#!/usr/bin/env python3
"""
Simple chat REPL on top of the HOS engine.

Talks to the `hos` CLI (one resident GPU process per turn), streams the reply,
and saves every conversation to ~/Documents/hos/conversations/*.json.

NOTE: HOS does not yet support the qwen35 hybrid-SSM architecture, so the
Qwen3.5-9B won't run here *yet*. When SSM support lands, just point MODEL at:
  ~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HOS = os.path.expanduser("~/.cargo/bin/hos")
# Override with: HOS_CHAT_MODEL=/path/to/model.gguf python3 chat.py
# `hos` auto-detects the architecture, so this works for BOTH families:
#   - standard transformers (Llama / Qwen2 / Mistral), e.g. Qwen2.5-7B
#   - the Qwen3.5 hybrid (SSM + attention), e.g. the 9B below (experimental, ~11 tok/s)
MODEL = os.path.expanduser(os.environ.get(
    "HOS_CHAT_MODEL",
    "~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
))

SYSTEM = "You are AUDIA, a concise, technical assistant built by Hueman. No emojis."
MAX_TOKENS = 256
TEMP = 0.7
TOP_P = 0.8
TOP_K = 40
HISTORY = 8  # how many past turns to feed back in

CHAT_DIR = Path(os.path.expanduser("~/Documents/hos/conversations"))
CHAT_DIR.mkdir(parents=True, exist_ok=True)


def build_prompt(history):
    convo = ""
    for m in history[-HISTORY:]:
        role = "User" if m["role"] == "user" else "AUDIA"
        convo += f"{role}: {m['text']}\n\n"
    return f"{SYSTEM}\n\n{convo}AUDIA:"


STOPS = ("AUDIA:", "User:", "Human:", "You:", "<|", "*end_of_text*")
HOLD = max(len(s) for s in STOPS)  # hold back this many chars so we never print a partial stop


def generate(prompt):
    cmd = [HOS, "--gpu", "--no-echo", "-m", MODEL, "-n", str(MAX_TOKENS),
           "--temp", str(TEMP), "--top-p", str(TOP_P), "--top-k", str(TOP_K), "-p", prompt]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
    )
    out, shown, stopped = "", 0, False
    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        out += ch
        hit = min((i for i in (out.find(s) for s in STOPS) if i != -1), default=-1)
        if hit != -1:  # model started a new turn — cut it off
            sys.stdout.write(out[shown:hit]); sys.stdout.flush()
            out, stopped = out[:hit], True
            proc.terminate()
            break
        safe = len(out) - HOLD
        if safe > shown:
            sys.stdout.write(out[shown:safe]); sys.stdout.flush(); shown = safe
    if not stopped:
        sys.stdout.write(out[shown:]); sys.stdout.flush()
    proc.wait()
    return out.strip()


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log = CHAT_DIR / f"chat_{ts}.json"
    history = []
    print("AUDIA (HOS) ready — type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "text": user})
        print("AUDIA: ", end="", flush=True)
        reply = generate(build_prompt(history))
        print("\n")
        history.append({"role": "audia", "text": reply})
        log.write_text(json.dumps(
            {"model": MODEL, "started": ts, "conversation": history}, indent=2
        ))
    print(f"Saved: {log}")


if __name__ == "__main__":
    main()
