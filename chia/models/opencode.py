"""opencode CLI LLM backend.

:class:`OpenCodeLLM` wraps the ``opencode`` CLI (https://opencode.ai) as an LLM
backend, opencode is provider-agnostic: the model is given as ``provider/model`` (e.g.
``anthropic/claude-sonnet-4-6``) and opencode runs its own server-side agentic
tool loop, so there is no client-side MCP loop here.

WARNING: experimental. Only exercised by the tests in
chia/models/tests/test_opencode.py (mocked unit tests, plus opt-in live tests).
Not validated in production. Auth is environment-driven: opencode uses its own
stored credentials (``opencode auth login``) or provider env vars.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import ray

from chia.base.ChiaFunction import ChiaFunction
from chia.base.llm_call import QueryResult, LLMCallBase

if TYPE_CHECKING:
    from chia.base.tools.ChiaTool import ChiaTool


# ---------------------------------------------------------------------------
# Exceptions
#
# A parallel taxonomy to claude.py. Kept separate so this module stands alone;
# each carries ``__reduce__`` for Ray serialization.
# ---------------------------------------------------------------------------


class OpenCodeError(Exception):
    """Base for all opencode CLI errors."""

    def __init__(
        self,
        node_id: str,
        error_type: str,
        exit_code: int = -1,
        raw_message: str = "",
    ):
        self.node_id = node_id
        self.error_type = error_type
        self.exit_code = exit_code
        self.raw_message = raw_message
        super().__init__(f"{error_type} on {node_id}: {raw_message[:200]}")

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.error_type, self.exit_code, self.raw_message),
        )

    def __reduce_ex__(self, protocol):
        # Every subclass's __reduce__ gives the constructor args; the per-call record that
        # OpenCodeLLM.prompt attaches (attempts, usage) rides along as state across Ray.
        r = self.__reduce__()
        extra = {k: self.__dict__[k] for k in ("attempts", "usage") if k in self.__dict__}
        return (r[0], r[1], extra) if extra else r


class RateLimitError(OpenCodeError):
    """The provider behind opencode reported a usage/rate limit."""

    def __init__(
        self,
        node_id: str,
        reset_time: datetime,
        raw_message: str = "",
        exit_code: int = -1,
    ):
        self.reset_time = reset_time
        super().__init__(node_id, "rate_limit", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.reset_time, self.raw_message, self.exit_code),
        )


class AuthenticationError(OpenCodeError):
    """opencode has no/invalid credentials for the selected provider."""

    def __init__(self, node_id: str, exit_code: int = -1, raw_message: str = ""):
        super().__init__(node_id, "authentication_failed", exit_code, raw_message)

    def __reduce__(self):
        return (self.__class__, (self.node_id, self.exit_code, self.raw_message))


class BillingError(OpenCodeError):
    """The provider account has a billing/payment problem."""

    def __init__(self, node_id: str, exit_code: int = -1, raw_message: str = ""):
        super().__init__(node_id, "billing_error", exit_code, raw_message)

    def __reduce__(self):
        return (self.__class__, (self.node_id, self.exit_code, self.raw_message))


class InvalidRequestError(OpenCodeError):
    """Malformed request — bad model string, invalid config, unknown agent, etc."""

    def __init__(self, node_id: str, exit_code: int = -1, raw_message: str = ""):
        super().__init__(node_id, "invalid_request", exit_code, raw_message)

    def __reduce__(self):
        return (self.__class__, (self.node_id, self.exit_code, self.raw_message))


class ServerError(OpenCodeError):
    """Transient provider/server-side failure (5xx, overloaded, connection)."""

    def __init__(
        self,
        node_id: str,
        exit_code: int = -1,
        raw_message: str = "",
        retry_after: Optional[int] = None,
    ):
        self.retry_after = retry_after
        super().__init__(node_id, "server_error", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.exit_code, self.raw_message, self.retry_after),
        )


class MaxOutputTokensError(OpenCodeError):
    """The response was truncated at the output token limit."""

    def __init__(
        self,
        node_id: str,
        exit_code: int = -1,
        raw_message: str = "",
        partial_text: str = "",
    ):
        self.partial_text = partial_text
        super().__init__(node_id, "max_output_tokens", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.exit_code, self.raw_message, self.partial_text),
        )


class UnknownOpenCodeError(OpenCodeError):
    """Unclassified opencode CLI error."""

    def __init__(
        self,
        node_id: str,
        exit_code: int = -1,
        raw_message: str = "",
        stderr: str = "",
    ):
        self.stderr = stderr
        super().__init__(node_id, "unknown", exit_code, raw_message)

    def __reduce__(self):
        return (
            self.__class__,
            (self.node_id, self.exit_code, self.raw_message, self.stderr),
        )


# ---------------------------------------------------------------------------
# Session-id parser
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"\bses_[A-Za-z0-9]+\b")


def parse_session_id(stdout: str) -> Optional[str]:
    """Pull the opencode session id out of ``run --format json`` stdout.

    Each event is ``{type, sessionID, part:{sessionID, ...}}``; the opening
    ``step_start`` line reliably carries it. Falls back to a regex scan if the
    JSON shape changes.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = event.get("sessionID") or (event.get("part") or {}).get("sessionID")
        if sid:
            return sid
    m = _SESSION_ID_RE.search(stdout)
    return m.group(0) if m else None


def parse_run_error(stdout: str) -> Optional[dict]:
    """Pull the first structured error out of ``run --format json`` stdout.

    opencode emits ``{"type":"error", "sessionID":..., "error":{name, data}}``
    events for failures that happen before/around the model request — notably an
    unknown model id, which surfaces as ``{"name":"UnknownError","data":{"message":
    "Model not found: ..."}}``. These never reach the session export's
    ``messages[].info.error`` because no assistant message is ever created, so
    the run stream is the *only* place they appear (confirmed against opencode
    1.15.13). Returns the first such ``error`` dict (``{name, data}``), or
    ``None``. Genuine provider errors (e.g. APIError 401) appear in *both* the run
    stream and the export; the export copy is richer (full ``responseHeaders``),
    so callers prefer it and use this only as a fallback.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "error":
            err = event.get("error")
            if isinstance(err, dict) and "name" in err:
                return err
    return None


# ---------------------------------------------------------------------------
# Custom model providers
# ---------------------------------------------------------------------------


@dataclass
class AdditionalModelProvider:
    """A custom model provider to inject into the opencode config's ``provider`` block.

    opencode is provider-agnostic: beyond its built-in providers you can declare a
    custom or self-hosted one — an OpenAI-compatible endpoint, a private gateway, a
    local vLLM/Ollama server, etc. — directly in the config under ``provider.<id>``.
    Each entry names the AI-SDK loader package (``npm``), a display ``name``, the SDK
    ``options`` (notably ``base_url`` / ``api_key``), and the ``models`` the provider
    serves. See https://opencode.ai/docs/providers (Custom providers) for the schema.

    Once declared, select one of its models by passing ``model="<id>/<model-id>"`` to
    :class:`OpenCodeLLM` — e.g. ``AdditionalModelProvider(id="my-vllm", ...)`` exposes
    its models as ``my-vllm/<model-id>``.

    Attributes:
        id: Provider key. Used both as the key under ``provider`` in the config and
            as the ``provider`` half of ``provider/model``. Must be unique.
        models: Either a list of model-id strings (each expands to a bare ``{}``
            entry) or a dict mapping model-id -> model config, e.g.
            ``{"name": ..., "limit": {...}, "cost": {...}, "options": {...}}``.
        npm: The AI-SDK package opencode loads to talk to this provider. Defaults to
            ``@ai-sdk/openai-compatible``, which fits any OpenAI-compatible endpoint.
        name: Human-readable display name (defaults to ``id`` when omitted).
        base_url: Endpoint URL; written to ``options.baseURL``. Optional if supplied
            via ``options`` instead.
        api_key: Credential; written to ``options.apiKey``. May be a literal secret
            or an opencode ``{env:NAME}`` template that opencode expands at runtime
            (preferred — keeps the secret out of the on-disk config file).
        options: Extra SDK options merged into the provider's ``options`` block
            (e.g. ``headers``, custom timeouts). On conflict these win over
            ``base_url`` / ``api_key``.
    """

    id: str
    models: Union[List[str], Dict[str, dict]]
    npm: str = "@ai-sdk/openai-compatible"
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    options: Optional[Dict[str, object]] = None

    def to_config_entry(self) -> dict:
        """Render this provider as an opencode ``provider.<id>`` config value."""
        options: dict = {}
        if self.base_url is not None:
            options["baseURL"] = self.base_url
        if self.api_key is not None:
            options["apiKey"] = self.api_key
        if self.options:
            options.update(self.options)

        # A list of ids -> bare ``{id: {}}`` entries; a dict is passed through.
        if isinstance(self.models, dict):
            models = dict(self.models)
        else:
            models = {model_id: {} for model_id in self.models}

        entry: dict = {"npm": self.npm, "name": self.name or self.id, "models": models}
        if options:
            entry["options"] = options
        return entry


# ---------------------------------------------------------------------------
# The local model gateway
# ---------------------------------------------------------------------------

LOCAL_PROVIDER_ID = "local"
LOCAL_BASE_URL_DEFAULT = "http://127.0.0.1:18100/v1"
LOCAL_CONTEXT_DEFAULT = 262144
# A step's output cap, reasoning included. opencode asks for min(this model limit,
# OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX), and ADIR sets the second term far higher, so this is the cap a
# step gets: 81,920 is Qwen's recommendation for hard reasoning (at 32,768 thinking ran out and a call
# produced no program).
LOCAL_OUTPUT_DEFAULT = 81920
# A request to the gateway has two limits, both the gateway's: its first token within 600 s of the gateway
# accepting it (queue wait included), then no silence of 300 s; a stream that keeps producing is never cut.
# opencode's own waits sit behind them, so the gateway's rule is what fires:
# * headers: the gateway holds them back until the first token, so an error before it (the first-token
#   timeout, a context overflow) reaches opencode as an HTTP status, which opencode acts on (it retries a
#   503/504 after its Retry-After, it compacts on a context overflow). The wait is declared to the gateway
#   in X-LLM-Header-Timeout-S, which then never sends headers early (it would after this less a minute).
LOCAL_HEADER_TIMEOUT_S = 720
# * chunks: the gateway ends a stream silent for 300 s itself (an error event, on which opencode sends the
#   step again); this is only a backstop should the gateway itself hang.
LOCAL_CHUNK_TIMEOUT_S = 360
# A local call has no call-level timeout: a step may stream 81,920 tokens for as long as it keeps producing,
# and the gateway ends a hung one. The caller's timeout_seconds is not applied to a local call (ADIR turns a
# run file's null or 0 into 1,200 s); CHIA_LOCAL_LLM_CALL_TIMEOUT_S (seconds, unset or 0: none) sets one.
# the effort names opencode's --variant takes for other providers; Qwen-style local models have no effort
# control, so each is declared (an undeclared variant would not resolve) and adds nothing to the request
LOCAL_VARIANTS = ("minimal", "low", "medium", "high", "xhigh", "max")
# the environment the local provider reads, captured where an OpenCodeLLM is built (the driver), so a call
# that runs on a Ray worker, whose environment is the raylet's, is configured exactly as the driver asked
LOCAL_ENV_PREFIXES = ("CHIA_LOCAL_LLM_", "CHIA_LLM_TAG")


def local_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k.startswith(LOCAL_ENV_PREFIXES) and v}


def local_token(env: Dict[str, str]) -> Optional[str]:
    """The gateway's token: ``CHIA_LOCAL_LLM_TOKEN``, else the content of ``CHIA_LOCAL_LLM_TOKEN_FILE``."""
    if env.get("CHIA_LOCAL_LLM_TOKEN"):
        return env["CHIA_LOCAL_LLM_TOKEN"].strip()
    path = env.get("CHIA_LOCAL_LLM_TOKEN_FILE")
    if not path:
        return None
    try:
        with open(os.path.expanduser(path)) as f:
            return f.read().strip() or None
    except OSError:
        return None


def local_call_timeout(env: Dict[str, str]) -> Optional[int]:
    """A local call's timeout: ``CHIA_LOCAL_LLM_CALL_TIMEOUT_S`` when set to a positive number, else None."""
    try:
        v = int(float(env.get("CHIA_LOCAL_LLM_CALL_TIMEOUT_S") or 0))
    except ValueError:
        v = 0
    return v if v > 0 else None


def check_local(env: Optional[Dict[str, str]] = None) -> List[str]:
    """What keeps a ``local`` call from being run as the evaluation needs it (empty when nothing does): a
    run tag, so the gateway's log attributes every token to a run, and the gateway's token. The plain loop
    calls this before a run starts, and :class:`OpenCodeLLM` when it is built, so a misconfigured run fails
    at once rather than call by call."""
    env = os.environ if env is None else env
    problems = []
    if not env.get("CHIA_LLM_TAG"):
        problems.append("CHIA_LLM_TAG is not set: every local call must name its run (the gateway's request "
                        "log attributes tokens by it)")
    if not local_token(env):
        problems.append("no gateway token: set CHIA_LOCAL_LLM_TOKEN_FILE to the gateway's token file "
                        "(chialu-a3eval: /data2/mhnie/chialu-home/llm/state/gateway.token) or CHIA_LOCAL_LLM_TOKEN")
    return problems


def local_provider(model_id: str, session_id: Optional[str] = None,
                   env: Optional[Dict[str, str]] = None) -> AdditionalModelProvider:
    """The ``local`` provider: an OpenAI-compatible gateway in front of self-hosted model servers.

    ``provider: local`` with ``model: <id>`` (opencode's ``local/<id>``) selects it with no provider block in
    a run file: :class:`OpenCodeLLM` declares it in every call's config. What differs between hosts comes
    from the environment (captured where the OpenCodeLLM is built, see :data:`LOCAL_ENV_PREFIXES`):

    * ``CHIA_LOCAL_LLM_BASE_URL``: the gateway's OpenAI base URL (default ``http://127.0.0.1:18100/v1``, the
      chialu-a3eval gateway on squire).
    * ``CHIA_LOCAL_LLM_TOKEN_FILE`` (or ``CHIA_LOCAL_LLM_TOKEN``): the gateway's token, sent as the provider's
      API key (``Authorization: Bearer``); squire is shared, and the gateway refuses a request without it.
    * ``CHIA_LOCAL_LLM_CONTEXT`` / ``CHIA_LOCAL_LLM_OUTPUT``: the context and per-step output limits opencode
      plans with (defaults 262144 / 81920). opencode compacts a session before the context limit less the
      output limit and asks for at most the output limit per step (see :data:`LOCAL_OUTPUT_DEFAULT`).
    * ``CHIA_LLM_TAG``: sent as ``X-LLM-Tag``, so the gateway's request log attributes tokens to a run.

    Each call carries its own ``X-LLM-Session`` header (``session_id``, recorded in the call's metadata and
    attempts as ``gateway_session``): the gateway keeps every step of the call on the backend that holds its
    prefix cache, and its log rows name the call. ``includeUsage`` makes the stream report token usage, which
    opencode needs for compaction and CHIA for the call's usage. opencode's header wait is
    :data:`LOCAL_HEADER_TIMEOUT_S`, declared to the gateway in ``X-LLM-Header-Timeout-S``, and its chunk wait
    :data:`LOCAL_CHUNK_TIMEOUT_S`, both behind the gateway's own limits (a first token within 600 s of its
    acceptance, then no silence of 300 s), which detect a hung request. The call itself has no timeout.
    """
    env = os.environ if env is None else env
    ctx = int(env.get("CHIA_LOCAL_LLM_CONTEXT") or LOCAL_CONTEXT_DEFAULT)
    out = int(env.get("CHIA_LOCAL_LLM_OUTPUT") or LOCAL_OUTPUT_DEFAULT)
    headers = {"X-LLM-Session": session_id or uuid.uuid4().hex,
               "X-LLM-Header-Timeout-S": str(LOCAL_HEADER_TIMEOUT_S)}
    if env.get("CHIA_LLM_TAG"):
        headers["X-LLM-Tag"] = env["CHIA_LLM_TAG"]
    model = {"name": model_id, "limit": {"context": ctx, "output": out}, "reasoning": True, "tool_call": True,
             "temperature": True, "attachment": False, "variants": {v: {} for v in LOCAL_VARIANTS}}
    return AdditionalModelProvider(
        id=LOCAL_PROVIDER_ID, name="Local LLM gateway", models={model_id: model},
        base_url=env.get("CHIA_LOCAL_LLM_BASE_URL") or LOCAL_BASE_URL_DEFAULT, api_key=local_token(env) or "none",
        options={"headers": headers, "includeUsage": True, "timeout": False,
                 "headerTimeout": LOCAL_HEADER_TIMEOUT_S * 1000, "chunkTimeout": LOCAL_CHUNK_TIMEOUT_S * 1000})


@dataclass
class OpenCodeQueryResult(QueryResult):
    """:class:`QueryResult` specialised for the opencode CLI.

    ``usage`` holds the run's summed metrics from ``opencode export`` (the same
    dict pushed to the profiler as ``_last_metadata``): ``input_tokens``,
    ``output_tokens``, ``reasoning_tokens``, ``cache_read``, ``cache_write``,
    ``cost_usd`` (from opencode's model price table) and ``num_turns``.
    ``session_id`` is opencode's id for the session the run created.
    """

    usage: Optional[dict] = None
    session_id: Optional[str] = None
    # One entry per attempt of this call (see OpenCodeLLM.prompt); ``usage`` is their sum.
    attempts: Optional[list] = None
    # The structured error event ({name, data}) of a failed call's last attempt.
    error: Optional[dict] = None
    # The X-LLM-Session the local gateway saw for this opencode run (``local`` models only): the key that
    # joins the call to the gateway's request log.
    gateway_session: Optional[str] = None


_USAGE_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read", "cache_write",
               "cost_usd", "num_turns")


def _sum_usage(attempts: List[dict]) -> dict:
    """The attempts' usage dicts summed key by key (numbers only)."""
    total: dict = {}
    for a in attempts:
        for k, v in (a.get("usage") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + v
    return total


def _public_attempts(attempts: List[dict]) -> List[dict]:
    """The attempt records without their transcripts (a failed result carries the last one)."""
    return [{k: v for k, v in a.items() if k != "stream"} for a in attempts]


def _failure_text(attempts: List[dict]) -> str:
    """Why a call failed: the last attempt's error type and message, its structured error event and
    its CLI stderr, after a one-line summary of the earlier attempts."""
    if not attempts:
        return "no attempt was made"
    lines = []
    for a in attempts[:-1]:
        lines.append(f"attempt {a['attempt']}: {a.get('error_type', 'ok')}: "
                     f"{str(a.get('error', ''))[:300]}")
    a = attempts[-1]
    lines.append(f"attempt {a['attempt']} of {len(attempts)} (last): {a.get('error_type', 'unknown')}: "
                 f"{a.get('error', '')}")
    if a.get("error_event"):
        lines.append("error event: " + json.dumps(a["error_event"], default=str)[:4000])
    if a.get("session_id"):
        lines.append(f"session: {a['session_id']}")
    if a.get("stderr"):
        lines.append("stderr: " + a["stderr"][-4000:])
    return "\n".join(lines)


_RATE_LIMIT_SIGNS = (
    "resource exhausted",            # vertex, project-level
    "rate limit",
    "rate-limited",
    "ratelimit",
    "too many requests",
    "429",
    "resource_exhausted",
    "quota exceeded",
)


def _is_rate_limit(name: str, message: str) -> bool:
    """Whether a provider error is back-pressure rather than a bad request.

    Read alongside the `statusCode` test at the call site: a provider that
    sends a status gets classified by it, and this catches the ones that
    only say so in words.
    """
    blob = f"{name} {message}".lower()
    return any(s in blob for s in _RATE_LIMIT_SIGNS)


_BILLING_SIGNS = (
    "insufficient balance",          # DeepSeek: HTTP 402 {"error":{"message":"Insufficient Balance"}}
    "insufficient_balance",
    "insufficient credit",
    "available credits",             # openrouter: "exceed your available credits"
    "credit balance is too low",     # anthropic
    "payment required",              # the 402 reason phrase
)
_HTTP_402_RE = re.compile(r"(?<![\d.])402(?![\d.])")


def _is_billing(message: str) -> bool:
    """Whether an error text names an out-of-credit / payment refusal.

    Only explicit phrases (and the 402 status as a whole number) count: a
    billing refusal must stop a run instead of being retried or counted as a
    bad request, and a false positive would stop a run that could go on.
    """
    blob = (message or "").lower()
    return any(s in blob for s in _BILLING_SIGNS)


_STDERR_402_RE = re.compile(r"(?:\bstatus(?:_?code)?|\bhttp(?:/[0-9.]+)?|\bcode|\berror|apierror)[^0-9a-z]{0,16}402\b|\b402\s+payment")


def _stderr_is_billing(stderr: str) -> bool:
    """A billing refusal in the CLI's plain-text stderr (no structured error):
    one of the phrases, or a 402 status next to the words 'status'/'http'/'code'."""
    blob = (stderr or "").lower()
    if _is_billing(blob):
        return True
    # 402 must sit next to a status word ("status 402", "HTTP/1.1 402", "code: 402", "402 Payment"): a stack
    # frame (index.js:402:17) or "402 tokens" beside the word "opencode" is not a billing refusal
    return bool(_STDERR_402_RE.search(blob))


def _rate_limit_waits() -> List[int]:
    """Seconds to wait between 429 retries, longest repeated at the end.

    Vertex answers a project-level rate limit for minutes at a time, so the
    ladder ends on a plateau rather than growing without bound.  Set
    CHIA_RATE_LIMIT_RETRIES to change how many waits are allowed (0 disables).
    """
    ladder = [30, 60, 120, 240, 300, 300]
    try:
        n = int(os.environ.get("CHIA_RATE_LIMIT_RETRIES", len(ladder)))
    except ValueError:
        n = len(ladder)
    n = max(0, n)
    if n <= len(ladder):
        return ladder[:n]
    return ladder + [ladder[-1]] * (n - len(ladder))


def opencode_tool_output_dir(env: Optional[dict] = None) -> str:
    """opencode's shared tool-output directory (``<data>/opencode/tool-output``).

    opencode writes a tool output that exceeds its truncation limit there, for every session of the
    user, and computes the directory as ``$XDG_DATA_HOME/opencode/tool-output`` (``~/.local/share``
    when XDG_DATA_HOME is unset, ``~`` being ``$HOME``) -- this mirrors that computation.
    """
    env = os.environ if env is None else env
    data = env.get("XDG_DATA_HOME") or os.path.join(
        env.get("HOME") or os.path.expanduser("~"), ".local", "share")
    return os.path.normpath(os.path.join(data, "opencode", "tool-output"))


def _deny_last(block, pattern: str):
    """A permission value (``"allow"``, ``{pattern: action}`` or absent) with ``pattern: deny`` as its
    last rule (opencode takes the last matching rule). An absent value gains only that rule, so
    opencode's defaults for the other paths stand."""
    if isinstance(block, dict):
        out = {k: v for k, v in block.items() if k != pattern}
    elif isinstance(block, str):
        out = {"*": block}
    else:
        out = {}
    out[pattern] = "deny"
    return out


def deny_tool_output(permission: dict, env: Optional[dict] = None) -> dict:
    """*permission* with opencode's shared tool-output directory denied.

    opencode appends ``external_directory: {<data>/opencode/tool-output/*: allow}`` after the configured
    rules (last rule wins) -- unless the agent's rules already carry an ``external_directory`` *deny* for
    exactly that glob. So the glob, spelled as opencode spells it, is denied here as the last
    external_directory rule, which both suppresses the append and wins over opencode's own default allow.
    The read tool also asks ``read`` with the path relative to the worktree, so a ``read`` deny on the
    directory backs that up (glob and grep ask with their search pattern, not the path: only the
    external_directory rule covers them).
    """
    perm = dict(permission or {})
    d = opencode_tool_output_dir(env)
    globs = [os.path.join(d, "*")]
    real = os.path.join(os.path.realpath(d), "*")
    if real not in globs:
        globs.append(real)
    ext = perm.get("external_directory")
    for g in globs:
        ext = _deny_last(ext, g)
    perm["external_directory"] = ext
    read = perm.get("read")
    for pat in ("*opencode/tool-output", "*opencode/tool-output/*"):
        read = _deny_last(read, pat)
    perm["read"] = read
    return perm


class OpenCodeLLM(LLMCallBase):
    """Wraps the ``opencode`` CLI as an LLM backend.

    Each :meth:`prompt` call runs ``opencode run`` (to create a session and get
    its id) then ``opencode export`` (to read the assistant response + usage
    from opencode's local DB). Returns the same :class:`QueryResult` shape as the
    other backends; ``returncode`` is the ``run`` exit code.
    """

    # Honors both --dangerously-skip-permissions and a `permission` config block.
    supports_dangerously_skip_permissions = True
    supports_config = True

    def __init__(
        self,
        model: Optional[str] = None,
        system_message: str = "",
        timeout_seconds: int = 600,
        retries: int = 3,
        logging_name: str = "opencode",
        logging_level: int = logging.DEBUG,
        log_dir: Optional[str] = None,
        opencode_bin: str = "opencode",
        agent_name: str = "chia",
        work_dir: Optional[str] = None,
        extra_cli_args: Optional[List[str]] = None,
        additional_providers: Optional[List[AdditionalModelProvider]] = None,
        dangerously_skip_permissions: bool = True,
        config: Optional[dict] = None,
    ):
        super().__init__(system_message=system_message,
                         dangerously_skip_permissions=dangerously_skip_permissions,
                         config=config)
        self.logging_level = logging_level
        self.logging_name = logging_name
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.opencode_bin = opencode_bin
        self.agent_name = agent_name
        self.work_dir = work_dir
        self.extra_cli_args = extra_cli_args or []
        self.additional_providers = additional_providers or []
        self.logger = logging.getLogger(logging_name)
        self._last_metadata: dict = {}
        self._last_export_error: Optional[dict] = None

        self.logger.warning(
            "OpenCodeLLM is experimental: only exercised by unit tests so far, "
            "not validated in production."
        )

        # opencode falls back to its own configured default model when none is
        # passed on the CLI, so model is optional here — just say so.
        if self.model is None:
            self.logger.info(
                "OpenCodeLLM: no model specified; opencode will use its "
                "configured default model."
            )

        # A `local/<model>` call goes to the local LLM gateway: its settings are taken here, where the node
        # is built (a call may run on a Ray worker with another environment), and a run that would not be
        # accounted (no tag, no token) is refused now. It runs without a call-level timeout (see
        # LOCAL_CHUNK_TIMEOUT_S): the gateway ends a hung request.
        self._local_env: Dict[str, str] = {}
        self._gateway_session: Optional[str] = None
        if (self.model or "").startswith(LOCAL_PROVIDER_ID + "/") and not any(
                p.id == LOCAL_PROVIDER_ID for p in self.additional_providers):
            self._local_env = local_env()
            problems = check_local(self._local_env)
            if problems:
                raise ValueError("OpenCodeLLM(%s): %s" % (self.model, "; ".join(problems)))
            self.timeout_seconds = local_call_timeout(self._local_env)

        self._log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_prefix = os.path.join(log_dir, f"{logging_name}_{run_id}")
        else:
            self._log_prefix = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # max_retries=0: Ray must never re-run a whole agent call on its own (a worker that dies mid-call
    # would otherwise start the session again, unseen and uncounted); the call fails instead. The
    # retries below are this method's own, and each one is recorded in the result's ``attempts``.
    @ChiaFunction(resources={"opencode_creds": 0.01}, max_retries=0, retry_exceptions=False)
    def prompt(
        self,
        user_message: str,
        tools: Optional[List[ChiaTool]] = [],
    ) -> QueryResult:
        """Send *user_message* to opencode and return the response.

        Every attempt (a rate-limit wait's retry included) is recorded: the result's ``attempts`` holds
        one ``{attempt, session_id, usage, error_type, error, error_event, returncode, stderr}`` per
        attempt, and its ``usage`` is the sum over all of them, so the tokens and cost of a failed or
        retried attempt are never lost. A typed error that propagates carries the same ``attempts`` and
        ``usage`` attributes.

        Returns:
            :class:`QueryResult` with ``success=True`` when opencode ran cleanly,
            or ``success=False`` when every retry attempt failed; then ``stderr``
            names the last attempt's failure (its error type and message, the
            structured error event and the CLI's stderr).

        Raises:
            RateLimitError / AuthenticationError / BillingError /
            InvalidRequestError: propagate immediately.
            ServerError: after all retries with exponential backoff.
            MaxOutputTokensError: after one retry attempt.
        """
        import time as _time

        from chia.trace.profiler import get_profiler

        profiler = get_profiler()

        # A 429 is the provider's back-pressure, not a bad call.  Upstream
        # propagates it immediately, which burns one of this call's attempts
        # (ADIR counts and bills those) and, on Vertex, throws the whole run
        # away during a rate-limit window.  Wait it out on a budget of its own
        # instead; CHIA_RATE_LIMIT_RETRIES=0 restores the upstream behaviour.
        rl_waits = _rate_limit_waits()
        rl_used = 0
        attempts: List[dict] = []

        def _raise(exc: Exception):
            exc.attempts = _public_attempts(attempts)
            exc.usage = _sum_usage(attempts)
            raise exc

        attempt = 0
        while attempt < self.retries:
            cli = None
            try:
                self._last_metadata = {}
                self._last_export_error = None
                cli = self._run_opencode(user_message, tools)
                self._last_metadata["model"] = self.model or "<opencode default>"
                if self._gateway_session:
                    self._last_metadata["gateway_session"] = self._gateway_session
                self._last_metadata["tools"] = [
                    {"name": t.name, "hostname": getattr(t, "hostname", None),
                     "port": getattr(t, "port", None),
                     "node_id": getattr(t, "node_id", None)}
                    for t in tools
                ]
                if profiler.enabled and self._last_metadata:
                    profiler.add_info(self._last_metadata)

                self._classify_error(
                    cli, export_error=getattr(self, "_last_export_error", None),
                )

                attempts.append(self._attempt_record(len(attempts) + 1, cli, None))
                cli.success = True
                cli.usage = _sum_usage(attempts) or cli.usage
                cli.attempts = _public_attempts(attempts)
                return cli

            except Exception as exc:  # noqa: BLE001 -- recorded, then dispatched by type below
                if cli is None:
                    cli = getattr(exc, "partial", None)   # a timed-out run's session, if it had one
                attempts.append(self._attempt_record(len(attempts) + 1, cli, exc))
                if self._gateway_session and "gateway_session" not in attempts[-1]:
                    attempts[-1]["gateway_session"] = self._gateway_session   # the gateway saw it all the same

                # -- Rate limit: wait it out without spending an attempt --
                if isinstance(exc, RateLimitError):
                    if rl_used >= len(rl_waits):
                        _raise(exc)
                    wait = rl_waits[rl_used]
                    # The provider's own reset time wins when it asks for longer than the ladder;
                    # RateLimitError carries it as a datetime, parsed from Retry-After where the
                    # provider sends one and an hour-ahead default where it does not, so it is
                    # capped rather than trusted outright.
                    reset = getattr(exc, "reset_time", None)
                    if reset is not None:
                        try:
                            hinted = (reset - datetime.now(timezone.utc)).total_seconds()
                            wait = max(wait, min(int(hinted), 900))
                        except (TypeError, ValueError, AttributeError):
                            pass
                    rl_used += 1
                    self.logger.warning(
                        "Rate limited (429); waiting %ds, then retrying (%d/%d waits used, "
                        "attempt %d/%d unspent)",
                        wait, rl_used, len(rl_waits), attempt + 1, self.retries,
                    )
                    _time.sleep(wait)
                    continue

                # -- Never retry: propagate immediately --
                if isinstance(exc, (AuthenticationError, BillingError, InvalidRequestError)):
                    _raise(exc)

                # -- Retry once: stochastic generation may produce shorter output --
                if isinstance(exc, MaxOutputTokensError):
                    if attempt == 0:
                        self.logger.warning(
                            "Max output tokens on attempt %d/%d, retrying once",
                            attempt + 1, self.retries,
                        )
                        attempt += 1
                        continue
                    _raise(exc)

                # -- Retry with exponential backoff: transient service issue --
                if isinstance(exc, ServerError):
                    backoff = min(5 * 2 ** attempt, 60)
                    self.logger.warning(
                        "Server error on attempt %d/%d, backing off %ds",
                        attempt + 1, self.retries, backoff,
                    )
                    if attempt + 1 < self.retries:
                        _time.sleep(backoff)
                elif isinstance(exc, UnknownOpenCodeError):
                    self.logger.warning(
                        "Unknown error on attempt %d/%d: %s",
                        attempt + 1, self.retries, exc,
                    )
                elif isinstance(exc, subprocess.TimeoutExpired):
                    self.logger.warning(
                        "Timeout on attempt %d/%d", attempt + 1, self.retries,
                    )
                else:
                    self.logger.warning(
                        "Unexpected error on attempt %d/%d: %s",
                        attempt + 1, self.retries, exc,
                    )

            attempt += 1

        last = attempts[-1] if attempts else {}
        return OpenCodeQueryResult(
            result="",
            returncode=last.get("returncode") if last.get("returncode") not in (None, 0) else -1,
            stderr=_failure_text(attempts),
            stream_result=last.get("stream", "") or "",
            success=False,
            usage=_sum_usage(attempts) or None,
            session_id=last.get("session_id"),
            attempts=_public_attempts(attempts),
            error=last.get("error_event"),
            gateway_session=last.get("gateway_session"),
        )

    @staticmethod
    def _attempt_record(n: int, cli, exc: Optional[BaseException]) -> dict:
        """One attempt as the result's ``attempts`` lists it (``stream`` is dropped from the list
        entries the caller sees except the last one's, which becomes a failed result's transcript)."""
        rec: dict = {"attempt": n, "session_id": getattr(cli, "session_id", None),
                     "usage": dict(getattr(cli, "usage", None) or {})}
        if getattr(cli, "gateway_session", None):
            rec["gateway_session"] = cli.gateway_session
        if cli is not None:
            rec["returncode"] = getattr(cli, "returncode", None)
            rec["stream"] = getattr(cli, "stream_result", "") or ""
            err = (getattr(cli, "stderr", "") or "").strip()
            if err:
                rec["stderr"] = err[-4000:]
            if getattr(cli, "error", None):
                rec["error_event"] = cli.error
        if exc is not None:
            rec["error_type"] = type(exc).__name__
            if isinstance(exc, subprocess.TimeoutExpired):
                rec["error"] = f"timed out after {exc.timeout} s"
            else:
                rec["error"] = (getattr(exc, "raw_message", "") or str(exc))[:4000]
            if isinstance(exc, UnknownOpenCodeError) and exc.stderr and not rec.get("stderr"):
                rec["stderr"] = exc.stderr.strip()[-4000:]
        return rec

    def _get_node_id(self) -> str:
        try:
            return ray.get_runtime_context().get_node_id()
        except Exception:
            return "unknown"

    def _classify_error(self, cli: QueryResult,
                        export_error: Optional[dict] = None) -> None:
        """Inspect *cli* and *export_error* and raise a typed error if wrong.

        ``opencode run`` almost always exits 0 even on failure (opencode bug
        #14551), so the exit code alone is not trustworthy. Logic:

        1. Clean run (exit 0, non-empty result, no structured error) -> return.
        2. Structured ``export_error`` (a ``{name, data}`` object from the
           session export or run stream) -> map to a typed error. This is the
           only reliable signal and the only thing we classify on.
        3. Any other failure (no structured error: a CLI/process-level failure
           on stderr, or an empty response) -> :class:`UnknownOpenCodeError`
           with the raw stderr attached. We deliberately do NOT keyword-match
           stderr: opencode emits its real errors as structured JSON (handled by
           (2)), so stderr only carries CLI/usage text, and guessing a type from
           it was imprecise — better to surface it honestly as unknown.

        Note the guard requires ``not export_error``: a structured error is
        honored even on an exit-0 run that returned partial text.
        """
        if cli.returncode == 0 and cli.result and not export_error:
            return

        node_id = self._get_node_id()

        # -- Path A: structured error from the session export (preferred) --
        if export_error:
            name = export_error.get("name", "")
            data = export_error.get("data", {}) or {}
            message = data.get("message", "") or ""
            status = data.get("statusCode")

            # Billing — out of credit (DeepSeek answers HTTP 402 "Insufficient Balance"). Checked
            # first: it must stop the caller, never be waited on, retried or taken for a bad request.
            if status == 402 or _is_billing(message):
                raise BillingError(node_id, cli.returncode, message or str(export_error))

            # Rate limit — honor the provider's Retry-After when present.
            #
            # `statusCode` alone misses most of them. Vertex answers a project-level limit with
            # `AI_APICallError: Resource exhausted. Please try again later.` and OpenRouter with
            # `is temporarily rate-limited upstream`, and neither reaches here with a status: when
            # the run fails before an assistant message exists the export holds no error at all and
            # the run stream's event carries a message and no code. Over one five-model comparison
            # the log held 23 such rate limits while this branch fired zero times, so every one of
            # them took the generic failure path -- a wasted attempt instead of a wait. Matching the
            # message as well is what makes the backoff above reachable.
            if status == 429 or _is_rate_limit(name, message):
                headers = data.get("responseHeaders", {}) or {}
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                reset_time = datetime.now(timezone.utc) + timedelta(seconds=60)
                if retry_after:
                    try:
                        reset_time = datetime.now(timezone.utc) + timedelta(
                            seconds=int(retry_after),
                        )
                    except (ValueError, TypeError):
                        pass
                raise RateLimitError(
                    node_id=node_id,
                    reset_time=reset_time,
                    raw_message=message,
                    exit_code=cli.returncode,
                )

            # Authentication.
            if name == "ProviderAuthError" or status in (401, 403):
                raise AuthenticationError(node_id, cli.returncode, message)

            # Billing / quota — APIError whose message names a billing problem.
            if name == "APIError" and message:
                if any(kw in message.lower() for kw in (
                    "billing", "quota", "payment", "credit", "subscription", "plan",
                )):
                    raise BillingError(node_id, cli.returncode, message)

            # Output token limit / context overflow.
            if name in ("ContextOverflowError", "MessageOutputLengthError"):
                raise MaxOutputTokensError(
                    node_id, cli.returncode, message, partial_text=cli.result,
                )

            # Server error (5xx or explicitly retryable) vs. invalid request.
            if name == "APIError":
                if (status and status >= 500) or data.get("isRetryable"):
                    raise ServerError(
                        node_id, exit_code=cli.returncode, raw_message=message,
                    )
                raise InvalidRequestError(node_id, cli.returncode, message)

            # Any other structured error (MessageAbortedError, UnknownError, ...).
            raise UnknownOpenCodeError(
                node_id, cli.returncode, message or str(export_error),
                stderr=cli.stderr,
            )

        # No structured error: a CLI/process-level failure (its message is on
        # stderr) or an empty response. opencode surfaces its real errors as
        # structured JSON (handled above), so there's nothing reliable to
        # classify here — report it honestly as unknown with the stderr attached.
        # The one exception is an explicit billing refusal: it must stop the caller.
        if _stderr_is_billing(cli.stderr):
            raise BillingError(node_id, cli.returncode, cli.stderr[-2000:])
        raise UnknownOpenCodeError(
            node_id, cli.returncode, cli.stderr[:300] or "empty response",
            stderr=cli.stderr,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_config(self, tools: List[ChiaTool]) -> dict:
        """Build the opencode config (written to OPENCODE_CONFIG).

        Defines the ``chia`` agent carrying our system prompt, one remote MCP
        server per ChiaTool, and any custom model providers passed at construction.
        """
        cfg: dict = {
            "$schema": "https://opencode.ai/config.json",
            "agent": {
                self.agent_name: {
                    "mode": "primary",
                    "prompt": self.system_message or "You are a helpful assistant.",
                }
            },
        }
        if tools:
            mcp: dict = {}
            for tool in tools:
                port = getattr(tool, "port", 8000)
                mcp[tool.name] = {
                    "type": "remote",
                    "url": f"http://{tool.hostname}:{port}/{tool.name}/mcp",
                    "enabled": True,
                }
            cfg["mcp"] = mcp

        # Config block. Defaults to allowing all tools — notably
        # external_directory, whose "ask" default blocks a non-interactive run
        # (the --dangerously-skip-permissions flag does NOT cover it).
        # Override via the `permission` kwarg.
        perm = self.config if self.config is not None else {
            "edit": "allow",
            "bash": "allow",
            "webfetch": "allow",
            "external_directory": "allow",
        }
        # opencode's tool-output directory is shared by every session of the user (other runs' truncated
        # tool outputs land there), so no call may read it, whatever the caller's block says.
        cfg["permission"] = deny_tool_output(perm)
        providers = list(self.additional_providers)
        prefix = LOCAL_PROVIDER_ID + "/"
        self._gateway_session = None
        if (self.model or "").startswith(prefix) and not any(p.id == LOCAL_PROVIDER_ID for p in providers):
            # a fresh session id per opencode run: one run is one agent session for the gateway's stickiness,
            # and the id joins this attempt to the gateway's request log
            self._gateway_session = uuid.uuid4().hex
            env = {**os.environ, **getattr(self, "_local_env", {})}
            providers.append(local_provider(self.model[len(prefix):], self._gateway_session, env))
        if providers:
            provider_cfg: dict = cfg.setdefault("provider", {})
            for provider in providers:
                provider_cfg[provider.id] = provider.to_config_entry()
        return cfg

    # Linux caps one argv string at MAX_ARG_STRLEN (128 KiB); a longer message fails with E2BIG
    # ("Argument list too long") before opencode starts. Above this size the message goes on stdin,
    # which ``opencode run`` reads when it is not a terminal.
    STDIN_MESSAGE_BYTES = 64 * 1024

    def _message_on_stdin(self, user_message: str) -> bool:
        return len(user_message.encode("utf-8")) > self.STDIN_MESSAGE_BYTES

    def _build_run_cmd(self, user_message: str) -> list:
        """Build the ``opencode run`` command list (message is a positional arg, or on stdin when long)."""
        cmd = [
            self.opencode_bin,
            "run",
            "--format", "json",
            "--agent", self.agent_name,
        ]
        if self.dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        if self.model:  # omit --model so opencode uses its configured default
            cmd += ["--model", self.model]
        if self.work_dir:
            cmd += ["--dir", self.work_dir]
        if self.extra_cli_args:
            cmd += self.extra_cli_args
        if not self._message_on_stdin(user_message):
            cmd.append(user_message)
        return cmd

    def _run_opencode(
        self,
        user_message: str,
        tools: Optional[List[ChiaTool]] = None,
    ) -> QueryResult:
        """Run ``opencode run`` then ``opencode export`` and assemble a QueryResult."""
        tools = tools or []
        cfg = self._build_config(tools)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="opencode_cfg_", delete=False
        )
        json.dump(cfg, tmp)
        tmp.close()
        cfg_path = tmp.name

        # opencode picks up our config via OPENCODE_CONFIG; disable project
        # config so a stray opencode.json in cwd can't shadow it. Stored
        # credentials / provider env vars in the inherited environment provide
        # auth (we don't touch them).
        env = dict(os.environ)
        env["OPENCODE_CONFIG"] = cfg_path
        env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"

        run_cmd = self._build_run_cmd(user_message)
        self.logger.info("Running: %s ...", " ".join(run_cmd[:6]))

        try:
            # Capture via a file, not a pipe: a large run stream would otherwise
            # be truncated at 64 KiB (see _capture / module docstring), which can
            # drop the trailing error events parse_run_error looks for.
            run = self._capture(run_cmd, env,
                                stdin_text=user_message if self._message_on_stdin(user_message) else None)
        except subprocess.TimeoutExpired as exc:
            # The killed run still created a session and spent tokens: read them back from the
            # export so the attempt's session id and usage are recorded (exc.partial).
            exc.partial = self._timed_out_partial(exc, env)
            raise
        finally:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass

        session_id = parse_session_id(run.stdout)
        # Errors that happen before an assistant message exists (e.g. unknown
        # model) only appear as `type:"error"` events in the run stream, never
        # in the export — capture them here so they can be classified too.
        run_error = parse_run_error(run.stdout)

        # A failed run (non-zero, or no session created) → return so the caller
        # classifies. Surface any run-stream error so it isn't lost. Don't
        # attempt an export without a session id.
        if run.returncode != 0 or session_id is None:
            if run.returncode != 0:
                self.logger.warning(
                    "opencode run exited %d: %s", run.returncode, run.stderr[:500]
                )
            self._last_export_error = run_error
            return OpenCodeQueryResult(
                result="",
                returncode=run.returncode if run.returncode != 0 else -1,
                stderr=run.stderr or "no session id in opencode output",
                stream_result=run.stdout,
                session_id=session_id,
                error=run_error,
                gateway_session=self._gateway_session,
            )

        export = self._run_export(session_id, env)
        final_text, meta, stream, export_error = self._extract_from_export(export)
        self._last_metadata = meta
        # Prefer the export's error (richer — full responseHeaders); fall back to
        # the run-stream error for pre-request failures the export never records.
        self._last_export_error = export_error or run_error

        if self._log_prefix is not None:
            # Best-effort only: prompt() may run on a remote worker whose
            # filesystem lacks the (driver-side) log_dir — a logging hiccup must
            # never fail an otherwise-successful call. The transcript is always
            # returned in stream_result regardless.
            try:
                os.makedirs(os.path.dirname(self._log_prefix), exist_ok=True)
                truncated = user_message[:500] + ("..." if len(user_message) > 500 else "")
                with open(f"{self._log_prefix}.log", "a") as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] session {session_id}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"[User Message]\n{truncated}\n\n")
                    f.write(stream)
                    f.write("-" * 80 + "\n\n")
            except OSError as exc:
                self.logger.warning(
                    "Could not write opencode log %s.log: %s", self._log_prefix, exc
                )

        return OpenCodeQueryResult(
            result=final_text,
            returncode=0,
            stderr=run.stderr,
            stream_result=stream,
            # A copy: prompt() extends _last_metadata (model, tools) afterwards
            # and usage must stay the pure token/cost totals.
            usage=dict(meta) if meta else None,
            session_id=session_id,
            error=self._last_export_error,
            gateway_session=self._gateway_session,
        )

    def _timed_out_partial(self, exc: subprocess.TimeoutExpired, env: dict) -> Optional["OpenCodeQueryResult"]:
        """What a timed-out ``run`` left: its session id (from the partial stream) and, through
        ``opencode export``, the usage and transcript so far. ``None`` when no session was created."""
        out = exc.output if isinstance(exc.output, str) else (
            exc.output.decode("utf-8", "replace") if exc.output else "")
        session_id = parse_session_id(out or "")
        if session_id is None:
            return None
        try:
            _text, meta, stream, err = self._extract_from_export(self._run_export(session_id, env))
        except Exception:  # noqa: BLE001 -- best effort: the id alone is still worth recording
            meta, stream, err = {}, "", None
        stderr = exc.stderr if isinstance(exc.stderr, str) else (
            exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
        return OpenCodeQueryResult(
            result="", returncode=-1, stderr=stderr or "", stream_result=stream or out,
            usage=dict(meta) if meta else None, session_id=session_id,
            error=err or parse_run_error(out or ""), gateway_session=self._gateway_session,
        )

    # `opencode export` of a finished session: a bounded local read, even for a call without a timeout
    EXPORT_TIMEOUT_S = 600

    def _capture(self, cmd: list, env: dict, stdin_text: Optional[str] = None,
                 timeout: Optional[float] = -1) -> SimpleNamespace:
        """Run *cmd* capturing stdout to a temp FILE and return it.

        Returns a ``SimpleNamespace(returncode, stdout, stderr)`` (the same shape
        ``subprocess.run`` would, so callers read ``.stdout`` etc. unchanged).

        Why a file instead of ``capture_output=True``: opencode truncates its
        stdout at the OS pipe buffer (64 KiB on Linux) and still exits 0 when
        stdout is a pipe, so large ``run`` streams / ``export`` payloads come back
        cut mid-JSON and unparseable. A regular file has no such limit. stderr is
        small, so it stays on a pipe. ``stdin=DEVNULL`` because ``run`` blocks on
        an open stdin pipe. ``subprocess.TimeoutExpired`` propagates to the caller.
        ``timeout`` is the call's (``timeout_seconds``) unless given; None or 0 means none.
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".out", prefix="opencode_out_", delete=False
        )
        out_path = tmp.name
        tmp.close()
        in_path = None
        if stdin_text is not None:
            # a file, not a pipe: the whole message is there at once and EOF follows it
            fd, in_path = tempfile.mkstemp(suffix=".txt", prefix="opencode_in_")
            with os.fdopen(fd, "w") as fh:
                fh.write(stdin_text)
        try:
            with open(out_path, "w") as out_fh, open(in_path or os.devnull, "r") as in_fh0:
                try:
                    proc = subprocess.run(
                        cmd,
                        stdin=in_fh0,
                        stdout=out_fh,
                        stderr=subprocess.PIPE,
                        text=True,
                        # the call's timeout; None or 0 is none (a local call has none)
                        timeout=(self.timeout_seconds if timeout == -1 else timeout) or None,
                        env=env,
                    )
                except subprocess.TimeoutExpired as exc:
                    # the stream so far (it names the session) instead of nothing
                    out_fh.flush()
                    try:
                        with open(out_path, "r") as part:
                            exc.output = part.read()
                    except OSError:
                        pass
                    raise
            with open(out_path, "r") as in_fh:
                stdout = in_fh.read()
            return SimpleNamespace(
                returncode=proc.returncode, stdout=stdout, stderr=proc.stderr or ""
            )
        finally:
            for p in (out_path, in_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def _run_export(self, session_id: str, env: dict) -> dict:
        """``opencode export <id>`` → parsed session JSON (``{}`` on failure)."""
        cmd = [self.opencode_bin, "export", session_id]
        try:
            proc = self._capture(cmd, env, timeout=self.timeout_seconds or self.EXPORT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.logger.warning("opencode export timed out for %s", session_id)
            return {}
        if proc.returncode != 0:
            self.logger.warning(
                "opencode export exited %d: %s", proc.returncode, proc.stderr[:300]
            )
            return {}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            self.logger.warning("opencode export returned non-JSON for %s", session_id)
            return {}

    def _extract_from_export(self, export: dict):
        """Pull final assistant text, usage metadata, a stream trace, and any error.

        Export shape::

            {info, messages: [{info:{role, tokens, cost, error}, parts: [...]}]}

        Parts: ``{type:"text", text}``, ``{type:"reasoning", text}``,
        ``{type:"tool", tool, state:{status, input, output}}``, plus
        ``step-start``/``step-finish`` — the latter carrying that model step's
        ``tokens``, ``cost`` and stop ``reason``, rendered as a ``[Usage]`` block
        so the transcript shows per-step metrics (same shape as the Antigravity
        backend's). The final answer is the text of the last assistant message;
        tokens/cost are summed across assistant messages and appended as a
        closing ``[Result]`` line.

        Returns:
            ``(text, metadata, stream, export_error)`` where *export_error* is the
            ``{name, data}`` dict from the first assistant message carrying an
            ``info.error`` (opencode's discriminated error object), or ``None``.
            This matters because ``opencode run`` almost always exits 0 even on
            failure (opencode bug #14551), so the exit code alone can't be trusted
            — the structured error in the export is the reliable signal.
        """
        stream_parts: list[str] = []
        meta = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                "cache_read": 0, "cache_write": 0, "cost_usd": 0.0, "num_turns": 0}
        last_assistant_text = ""
        export_error = None

        for msg in export.get("messages", []) or []:
            info = msg.get("info", {}) if isinstance(msg, dict) else {}
            role = info.get("role")
            parts = msg.get("parts", []) if isinstance(msg, dict) else []

            if role == "assistant":
                if export_error is None:
                    err = info.get("error")
                    if isinstance(err, dict) and "name" in err:
                        export_error = err

                meta["num_turns"] += 1
                tok = info.get("tokens") or {}
                meta["input_tokens"] += tok.get("input", 0) or 0
                meta["output_tokens"] += tok.get("output", 0) or 0
                meta["reasoning_tokens"] += tok.get("reasoning", 0) or 0
                cache = tok.get("cache") or {}
                meta["cache_read"] += cache.get("read", 0) or 0
                meta["cache_write"] += cache.get("write", 0) or 0
                meta["cost_usd"] += info.get("cost", 0) or 0

                turn_text: list[str] = []
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    ptype = p.get("type")
                    if ptype == "text":
                        txt = p.get("text", "")
                        turn_text.append(txt)
                        stream_parts.append(f"[Response]\n{txt}\n\n")
                    elif ptype == "reasoning":
                        stream_parts.append(f"[Thinking]\n{p.get('text', '')}\n\n")
                    elif ptype == "tool":
                        state = p.get("state") or {}
                        args = json.dumps(state.get("input", {}))
                        if len(args) > 2000:
                            args = args[:2000] + "\n... [truncated]"
                        stream_parts.append(
                            f"[Tool Call: {p.get('tool', 'unknown')}]\nArgs: {args}\n\n"
                        )
                        out = state.get("output", "")
                        if not isinstance(out, str):
                            out = json.dumps(out)
                        if len(out) > 2000:
                            out = out[:2000] + "\n... [truncated]"
                        if out:
                            stream_parts.append(f"[Tool Result]\n{out}\n\n")
                    elif ptype == "step-finish":
                        step = dict(p.get("tokens") or {})
                        if p.get("cost") is not None:
                            step["cost_usd"] = p["cost"]
                        if p.get("reason"):
                            step["reason"] = p["reason"]
                        if step:
                            stream_parts.append(f"[Usage]\n{json.dumps(step)}\n\n")
                if turn_text:
                    last_assistant_text = "".join(turn_text)

        meta = {k: v for k, v in meta.items() if v}
        if meta:
            stream_parts.append(f"[Result]\n{json.dumps(meta)}\n\n")
        return last_assistant_text, meta, "".join(stream_parts), export_error
