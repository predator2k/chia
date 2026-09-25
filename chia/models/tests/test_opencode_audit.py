"""OpenCodeLLM: the shared tool-output directory, billing refusals, failure reasons and attempt records.

Offline (fake ``subprocess.run``), like the mocked tests of test_opencode.py::

    pytest chia/models/tests/test_opencode_audit.py -v
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import time
from types import SimpleNamespace

import pytest

from chia.models import opencode as oc_mod
from chia.models.opencode import (
    BillingError,
    OpenCodeLLM,
    QueryResult,
    RateLimitError,
    UnknownOpenCodeError,
    deny_tool_output,
    opencode_tool_output_dir,
)


def _step_start(sid):
    return json.dumps({"type": "step_start", "sessionID": sid,
                       "part": {"sessionID": sid, "type": "step-start"}})


def _export(text="PONG", cost=0.01, tokens=None, error=None):
    tokens = tokens or {"input": 10, "output": 4}
    info = {"role": "assistant", "tokens": tokens, "cost": cost}
    if error is not None:
        info["error"] = error
    parts = [{"type": "text", "text": text}] if text else []
    return {"info": {}, "messages": [{"info": {"role": "user"}, "parts": []},
                                     {"info": info, "parts": parts}]}


def _script(monkeypatch, steps):
    """Fake subprocess.run: *steps* is a list of (run_stdout, run_stderr, run_rc, export_obj), one per
    attempt; a step may instead be ("timeout", partial_stdout, export_obj)."""
    it = iter(steps)
    calls = []
    cur = {}

    def fake_run(cmd, **kw):
        sub = cmd[1]
        out = kw.get("stdout")
        calls.append(sub)
        if sub == "run":
            cur["step"] = next(it)
            st = cur["step"]
            if st[0] == "timeout":
                out.write(st[1])
                raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
            out.write(st[0])
            return SimpleNamespace(returncode=st[2], stdout=None, stderr=st[1])
        st = cur["step"]
        payload = json.dumps(st[-1])
        out.write(payload)
        return SimpleNamespace(returncode=0, stdout=None, stderr="")

    monkeypatch.setattr(oc_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return calls


# ---------------------------------------------------------------- tool-output directory

def test_tool_output_dir_follows_xdg_and_home():
    assert opencode_tool_output_dir({"XDG_DATA_HOME": "/x/share"}) == "/x/share/opencode/tool-output"
    assert opencode_tool_output_dir({"HOME": "/h"}) == "/h/.local/share/opencode/tool-output"


@pytest.mark.parametrize("ext", [None, "deny", "allow", {"*": "deny"}])
def test_tool_output_glob_is_the_last_external_directory_rule(ext):
    """opencode appends `external_directory: {<data>/opencode/tool-output/*: allow}` unless the agent
    already denies exactly that glob; the deny must be the last rule so it also beats the default allow."""
    env = {"XDG_DATA_HOME": "/x/share"}
    perm = {"edit": "deny"} if ext is None else {"edit": "deny", "external_directory": ext}
    out = deny_tool_output(perm, env)
    rules = out["external_directory"]
    assert list(rules.items())[-1] == ("/x/share/opencode/tool-output/*", "deny")
    if isinstance(ext, str):
        assert rules["*"] == ext                       # the caller's catch-all stands, first
    if ext is None:
        assert "*" not in rules                        # opencode's own defaults stand for other paths
    assert out["read"]["*opencode/tool-output/*"] == "deny"
    assert out["edit"] == "deny"
    assert perm.get("read") is None                    # the caller's dict is not mutated


def test_build_config_denies_tool_output_on_every_block(monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/x/share")
    for config in (None, {"edit": "allow", "external_directory": "deny", "read": {"*": "allow"}}):
        perm = OpenCodeLLM(config=config)._build_config([])["permission"]
        assert list(perm["external_directory"].items())[-1] == ("/x/share/opencode/tool-output/*", "deny")
        assert list(perm["read"].items())[-1][1] == "deny"
    perm = OpenCodeLLM(config={"read": {"*": "allow"}})._build_config([])["permission"]
    assert list(perm["read"])[0] == "*"                # deny patterns come after the catch-all


# ---------------------------------------------------------------- billing (402 / Insufficient Balance)

def _cli(stderr="", rc=0):
    return QueryResult(result="", returncode=rc, stderr=stderr, stream_result="")


@pytest.mark.parametrize("error", [
    # DeepSeek's out-of-balance reply as opencode's APIError
    {"name": "APIError", "data": {"message": "Insufficient Balance", "statusCode": 402, "isRetryable": False}},
    {"name": "APIError", "data": {"message": "Insufficient Balance", "isRetryable": False}},
    {"name": "APIError", "data": {"message": "Payment Required", "statusCode": 402}},
    {"name": "APIError", "data": {"message": "", "statusCode": 402}},
    # the same as a run-stream event without a status
    {"name": "UnknownError", "data": {"message": "AI_APICallError: Insufficient Balance"}},
    {"name": "APIError", "data": {"message": "This request requires more credits: exceed your available credits"}},
])
def test_classify_billing_payloads(error):
    with pytest.raises(BillingError):
        OpenCodeLLM()._classify_error(_cli(), export_error=error)


@pytest.mark.parametrize("stderr", [
    'Error: {"error":{"message":"Insufficient Balance","type":"unknown_error"}}',
    "AI_APICallError: HTTP status 402",
    "payment required: add credit",
])
def test_classify_billing_stderr(stderr):
    with pytest.raises(BillingError):
        OpenCodeLLM()._classify_error(_cli(stderr=stderr, rc=1))


@pytest.mark.parametrize("stderr", ["read 4020 bytes", "line 402 of cfg.json", "something weird"])
def test_classify_402_lookalikes_are_not_billing(stderr):
    with pytest.raises(UnknownOpenCodeError):
        OpenCodeLLM()._classify_error(_cli(stderr=stderr, rc=1))


def test_prompt_billing_stops_at_once_with_usage(monkeypatch):
    err = {"name": "APIError", "data": {"message": "Insufficient Balance", "statusCode": 402}}
    calls = _script(monkeypatch, [(_step_start("ses_b1"), "", 0, _export(text="", cost=0.002, error=err))] * 3)
    with pytest.raises(BillingError) as ei:
        OpenCodeLLM(retries=3).prompt("hi", tools=[])
    assert calls == ["run", "export"]                  # no retry
    assert ei.value.attempts[0]["session_id"] == "ses_b1"
    assert ei.value.usage["cost_usd"] == pytest.approx(0.002)
    # the record survives pickling (Ray ships the exception back to the caller)
    back = pickle.loads(pickle.dumps(ei.value))
    assert isinstance(back, BillingError) and back.attempts == ei.value.attempts
    assert back.usage == ei.value.usage and back.raw_message == "Insufficient Balance"


# ---------------------------------------------------------------- the failure reason (#12)

def test_failed_call_carries_the_real_reason(monkeypatch):
    doom = {"name": "UnknownError", "data": {"message": "The user has specified a rule which prevents you "
            "from using this specific tool call. [{\"permission\":\"doom_loop\",\"action\":\"deny\"}]"}}
    _script(monkeypatch, [(_step_start("ses_d1"), "some cli warning", 0, _export(text="half", cost=0.03,
                                                                                 error=doom))])
    res = OpenCodeLLM(retries=1).prompt("hi", tools=[])
    assert res.success is False
    assert "UnknownOpenCodeError" in res.stderr and "doom_loop" in res.stderr
    assert "some cli warning" in res.stderr and "ses_d1" in res.stderr
    assert res.error == doom and res.session_id == "ses_d1"
    assert res.usage["cost_usd"] == pytest.approx(0.03)
    assert "[Response]\nhalf" in res.stream_result     # the failed attempt's transcript


def test_failed_run_without_session_keeps_stderr(monkeypatch):
    _script(monkeypatch, [("", "boom: no provider configured", 1, {})])
    res = OpenCodeLLM(retries=1).prompt("hi", tools=[])
    assert res.success is False and "boom: no provider configured" in res.stderr
    assert res.returncode == 1


def test_run_stream_error_event_is_in_the_reason(monkeypatch):
    ev = json.dumps({"type": "error", "sessionID": "ses_e1",
                     "error": {"name": "UnknownError", "data": {"message": "Model not found: x/y"}}})
    _script(monkeypatch, [(_step_start("ses_e1") + "\n" + ev, "", 0, {"messages": []})])
    res = OpenCodeLLM(retries=1).prompt("hi", tools=[])
    assert "Model not found: x/y" in res.stderr and res.error["name"] == "UnknownError"


# ---------------------------------------------------------------- attempts and Ray retries (#13)

def test_prompt_is_never_retried_by_ray():
    opts = OpenCodeLLM.prompt._chia_options
    assert opts["max_retries"] == 0 and opts["retry_exceptions"] is False


def test_every_attempt_is_counted(monkeypatch):
    rl = {"name": "APIError", "data": {"message": "Rate Limited", "statusCode": 429}}
    doom = {"name": "UnknownError", "data": {"message": "doom_loop"}}
    calls = _script(monkeypatch, [
        (_step_start("ses_a1"), "", 0, _export(text="", cost=0.001, error=rl)),     # rate limit: waited out
        (_step_start("ses_a2"), "", 0, _export(text="x", cost=0.02, error=doom)),   # unknown: an attempt
        (_step_start("ses_a3"), "", 0, _export(text="PONG", cost=0.05)),            # success
    ])
    res = OpenCodeLLM(retries=2).prompt("hi", tools=[])
    assert res.success is True and res.result == "PONG" and res.session_id == "ses_a3"
    assert [a["session_id"] for a in res.attempts] == ["ses_a1", "ses_a2", "ses_a3"]
    assert [a.get("error_type") for a in res.attempts] == ["RateLimitError", "UnknownOpenCodeError", None]
    assert res.usage["cost_usd"] == pytest.approx(0.071)
    assert res.usage["input_tokens"] == 30 and res.usage["num_turns"] == 3
    assert calls.count("run") == 3
    assert all("stream" not in a for a in res.attempts)


def test_timed_out_attempt_keeps_its_session_and_usage(monkeypatch):
    _script(monkeypatch, [("timeout", _step_start("ses_t1") + "\n", _export(text="so far", cost=0.04))])
    res = OpenCodeLLM(retries=1, timeout_seconds=5).prompt("hi", tools=[])
    assert res.success is False and res.session_id == "ses_t1"
    assert res.usage["cost_usd"] == pytest.approx(0.04)
    assert "TimeoutExpired" in res.stderr and "timed out after 5" in res.stderr
    assert res.attempts[0]["session_id"] == "ses_t1"
