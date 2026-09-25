"""Actor and task lifetimes on a private local Ray instance (never a shared cluster).

* the cache actor of a driver without a namespace goes away with the driver
  (it used to be detached in an anonymous namespace: one leaked actor, worker
  process and head worker port per driver);
* with a fixed namespace, later drivers reuse the one detached actor;
* ``OpenCodeLLM.prompt`` is never re-executed by Ray when its worker dies
  (``max_retries=0``), and a typed opencode error keeps its attempt record
  across ``ray.get``;
* the dispatch proxy honours ``max_retries=0`` with ``max_task_retries=0``.

Run::

    TMPDIR=/some/scratch pytest chia/base/test/test_local_ray_lifetimes.py -v

The Ray temp dir is created under ``$TMPDIR`` (keep that path short: Ray's
socket paths must stay under 107 characters).
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest
import ray

from chia.base import cache as cache_mod


@pytest.fixture(scope="module")
def local_ray():
    os.environ.pop("RAY_ADDRESS", None)
    tmp = tempfile.mkdtemp(prefix="rt_")
    ray.init(address="local", num_cpus=2, resources={"opencode_creds": 1}, include_dashboard=False,
             _temp_dir=tmp, log_to_driver=False)
    gcs = ray.get_runtime_context().gcs_address
    try:
        yield SimpleEnv(gcs=gcs, tmp=tmp)
    finally:
        ray.shutdown()


class SimpleEnv:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _driver(gcs: str, body: str) -> str:
    """Run *body* in a separate driver process connected to the local instance; return its stdout."""
    code = textwrap.dedent(f"""
        import ray
        ray.init(address={gcs!r}, log_to_driver=False)
        from chia.base.cache import start_cache, get_active_cache
    """) + textwrap.dedent(body)
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def _actor_alive(name: str, namespace: str) -> bool:
    try:
        ray.get_actor(name, namespace=namespace)
        return True
    except ValueError:
        return False


def test_cache_without_namespace_dies_with_its_driver(local_ray, tmp_path):
    out = _driver(local_ray.gcs, f"""
        h = start_cache(size=1, units="MB", cache_dir_path={str(tmp_path / 'c1')!r})
        assert get_active_cache() is not None
        print(ray.get_runtime_context().namespace)
    """)
    ns = out.strip().splitlines()[-1]
    deadline = time.time() + 30
    while _actor_alive(cache_mod._CACHE_ACTOR_NAME, ns) and time.time() < deadline:
        time.sleep(0.5)
    assert not _actor_alive(cache_mod._CACHE_ACTOR_NAME, ns), "cache actor outlived its driver"


def test_cache_with_fixed_namespace_is_reused(local_ray, tmp_path):
    ns = "chia_cache_lifetime_test"
    ids = [
        _driver(local_ray.gcs, f"""
            h = start_cache(size=1, units="MB", cache_dir_path={str(tmp_path / 'c2')!r}, namespace={ns!r})
            print(ray.get_actor("ChiaCacheStore", namespace={ns!r})._actor_id.hex())
        """).strip().splitlines()[-1]
        for _ in range(2)
    ]
    assert ids[0] == ids[1]                     # one actor, reused by the second driver
    assert _actor_alive(cache_mod._CACHE_ACTOR_NAME, ns)
    cache_mod.stop_cache(ns)
    deadline = time.time() + 30
    while _actor_alive(cache_mod._CACHE_ACTOR_NAME, ns) and time.time() < deadline:
        time.sleep(0.5)
    assert not _actor_alive(cache_mod._CACHE_ACTOR_NAME, ns)


def test_prompt_is_not_rerun_when_its_worker_dies(local_ray, tmp_path):
    """A fake opencode that counts its runs and kills the Ray worker running prompt: with
    max_retries=0 the call fails once; Ray's default (3 retries) would have run it 4 times."""
    from chia.base.ChiaFunction import get
    from chia.models.opencode import OpenCodeLLM

    counter = tmp_path / "runs"
    fake = tmp_path / "opencode"
    fake.write_text(f"#!/bin/sh\necho run >> {counter}\nkill -9 $PPID\nsleep 5\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    llm = OpenCodeLLM(opencode_bin=str(fake), retries=1, timeout_seconds=60)
    with pytest.raises(Exception) as ei:
        get(llm.prompt.chia_remote(llm, "hi"))
    assert "WorkerCrashed" in type(ei.value).__name__ or "died" in str(ei.value).lower()
    assert counter.read_text().count("run") == 1


def test_billing_error_keeps_its_record_across_ray(local_ray):
    from chia.models.opencode import BillingError

    @ray.remote(max_retries=0)
    def fail():
        e = BillingError("node", 0, "Insufficient Balance")
        e.attempts = [{"attempt": 1, "session_id": "ses_x", "usage": {"cost_usd": 0.01}}]
        e.usage = {"cost_usd": 0.01}
        raise e

    with pytest.raises(BillingError) as ei:
        ray.get(fail.remote())
    assert ei.value.attempts[0]["session_id"] == "ses_x"
    assert ei.value.usage == {"cost_usd": 0.01}


def test_proxy_submit_honours_max_retries_zero(monkeypatch):
    """No Ray needed: the relayed dispatch carries max_task_retries=0 for a max_retries=0 function."""
    from chia.base import ChiaFunction as cf
    from chia.base import dispatch_proxy as dp

    seen = {}

    class Submit:
        def options(self, **kw):
            seen["options"] = kw
            return self

        def remote(self, *a, **kw):
            seen["remote"] = a
            return "ref"

    monkeypatch.setattr(dp, "should_proxy", lambda opts=None: True)
    monkeypatch.setattr(dp, "get_dispatch_proxy", lambda: type("P", (), {"submit": Submit()})())
    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: type("Q", (), {"enabled": False})())

    def f():
        return 1

    assert cf._try_proxy(f, {"max_retries": 0}, None, None, (), {}, None) == "ref"
    assert seen["options"] == {"max_task_retries": 0}
    seen.clear()
    cf._try_proxy(f, {"num_cpus": 1}, None, None, (), {}, None)
    assert "options" not in seen
