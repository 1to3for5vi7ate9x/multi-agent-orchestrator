"""Tests for core/session.py: persistence, ledger, rotation, memory."""
import json
import os
import stat
import sys
import tempfile
from pathlib import Path



from ml_orchestrator.core.session import (
    AntigravitySessionAdapter, ClaudeSessionAdapter, ManagedSession,
    MemoryStore, LATEST, estimate_tokens,
)

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")

# ---- adapters: command construction -----------------------------------------
print("== adapters ==")
ca = ClaudeSessionAdapter(["claude", "--permission-mode", "acceptEdits", "-p"])
cmd = ca.command("hi", None)
check("claude new cmd",
      cmd == ["claude", "--permission-mode", "acceptEdits",
              "--output-format", "json", "-p", "hi"], cmd)
cmd = ca.command("hi", "abc-123")
check("claude resume cmd", "--resume" in cmd and "abc-123" in cmd
      and cmd[-1] == "hi", cmd)

r = ClaudeSessionAdapter.parse(json.dumps({
    "result": "done", "session_id": "s1",
    "usage": {"input_tokens": 100, "cache_read_input_tokens": 50,
              "cache_creation_input_tokens": 25, "output_tokens": 10}}))
check("claude parse", r.text == "done" and r.session_id == "s1"
      and r.context_tokens == 185, r)
r = ClaudeSessionAdapter.parse("plain text, not json")
check("claude parse fallback", r.text == "plain text, not json"
      and r.context_tokens is None)

aa = AntigravitySessionAdapter(["agy", "-p"])
check("agy new cmd", aa.command("hi", None) == ["agy", "-p", "hi"])
check("agy continue cmd", aa.command("hi", LATEST) == ["agy", "-c", "-p", "hi"])
check("agy conv cmd",
      aa.command("hi", "c9") == ["agy", "--conversation", "c9", "-p", "hi"])
r = AntigravitySessionAdapter.parse("verdict text\n")
check("agy parse", r.text == "verdict text" and r.session_id == LATEST
      and r.context_tokens is None)

# ---- memory store -------------------------------------------------------------
print("== memory store ==")
with tempfile.TemporaryDirectory() as td:
    ms = MemoryStore(Path(td) / "mem")
    check("no snapshot yet", ms.load_snapshot("editor") is None
          and ms.render_preamble("editor") == "")
    ms.save_snapshot("editor", "# MEMORY SNAPSHOT\nfacts here", "test", 5000)
    check("snapshot saved", "facts here" in ms.load_snapshot("editor"))
    pre = ms.render_preamble("editor")
    check("preamble wraps", "<memory>" in pre and "facts here" in pre
          and "RESTORED MEMORY" in pre)
    ms.save_snapshot("editor", "# MEMORY SNAPSHOT v2", "test2", 9000)
    archive = (Path(td) / "mem" / "editor_memory_archive.jsonl").read_text()
    check("archive grows", archive.count("MEMORY SNAPSHOT") == 2)
    ms.save_state("editor", {"session_id": "s1", "context_tokens": 42})
    check("state roundtrip", ms.load_state("editor")["context_tokens"] == 42)

# ---- managed session with a scripted fake CLI ----------------------------------
print("== managed session ==")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    calls_log = td / "calls.jsonl"
    fake = td / "fakeclaude"
    script = """#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
prompt = args[-1]
resumed = "--resume" in args
with open(%r, "a") as f:
    f.write(json.dumps({"resumed": resumed, "prompt": prompt[:200]}) + "\\n")
if "MEMORY SNAPSHOT" in prompt:
    text = "# MEMORY SNAPSHOT\\n## Goal\\ndistilled facts"
    usage = 1000
else:
    text = "reply ok"
    usage = 600000  # huge: immediately past a 50%% threshold of 1M
print(json.dumps({"result": text, "session_id": "fake-sess-1",
                  "usage": {"input_tokens": usage, "output_tokens": 100}}))
""" % str(calls_log)
    fake.write_text(script)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    ms = MemoryStore(td / "mem")
    sess = ManagedSession(
        agent_name="editor", role_description="the Editor",
        adapter=ClaudeSessionAdapter([str(fake), "-p"]),
        memory=ms, cwd=td, context_limit=1_000_000, rotate_at=0.5,
        timeout=30,
    )
    out = sess.send("first prompt")
    check("send works", out == "reply ok", out)
    check("session id captured", sess.session_id == "fake-sess-1")
    check("real usage tracked", sess.context_tokens == 600100,
          sess.context_tokens)
    check("should rotate", sess.should_rotate())

    # second send resumes with --resume
    sess.send("second prompt")
    lines = [json.loads(l) for l in calls_log.read_text().splitlines()]
    check("second call resumed", lines[1]["resumed"] is True)

    snap = sess.rotate("unit test")
    check("rotation snapshot", snap and "distilled facts" in snap)
    check("session reset", sess.session_id is None
          and sess.context_tokens == 0 and sess.rotations == 1)
    check("snapshot persisted",
          "distilled facts" in ms.load_snapshot("editor"))

    # next send: fresh session, memory preamble injected
    sess.send("third prompt after rotation")
    lines = [json.loads(l) for l in calls_log.read_text().splitlines()]
    check("fresh after rotation", lines[-1]["resumed"] is False)
    check("memory injected", "RESTORED MEMORY" in lines[-1]["prompt"],
          lines[-1]["prompt"][:80])

    # cross-run persistence: a new ManagedSession picks up where we left off
    sess2 = ManagedSession(
        agent_name="editor", role_description="the Editor",
        adapter=ClaudeSessionAdapter([str(fake), "-p"]),
        memory=ms, cwd=td, context_limit=1_000_000, rotate_at=0.5,
        timeout=30,
    )
    check("state restored across runs",
          sess2.session_id == "fake-sess-1"
          and sess2.context_tokens > 0, sess2.status_line())

    # estimator path (adapter reporting no usage)
    sess3 = ManagedSession(
        agent_name="evaluator", role_description="the Evaluator",
        adapter=AntigravitySessionAdapter([str(fake), "-p"]),
        memory=ms, cwd=td, context_limit=1000, rotate_at=0.5, timeout=30,
    )
    sess3.send("x" * 4000)  # ~1000 est. prompt tokens > 50% of 1000
    check("estimator accumulates", sess3.context_tokens >= 1000,
          sess3.context_tokens)
    check("estimator triggers rotation", sess3.should_rotate())

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
