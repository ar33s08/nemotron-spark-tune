#!/usr/bin/env python3
"""Benchmark a vLLM OpenAI-compatible endpoint.

Measures single-stream chat + code throughput and aggregate concurrency scaling.
n=3 median per metric. Writes JSON so runs are comparable and diffable.

Usage:
    python3 bench.py --label tuned --base http://127.0.0.1:18300
"""
import argparse
import json
import statistics as st
import threading
import time
import urllib.request

CHAT = "Explain what a hash table is."
CODE = "Write a python function that merges two sorted lists."
CONC = "Count slowly from 1 to 40, one per line."


def run(base, prompt, mt, model, n=3):
    """Return median tok/s over n runs."""
    out = []
    for _ in range(n):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        j = json.load(urllib.request.urlopen(req, timeout=300))
        dt = time.time() - t0
        out.append(j.get("usage", {}).get("completion_tokens", 0) / max(dt, 1e-9))
    return st.median(out)


def conc(base, n, model):
    """Aggregate tok/s with n parallel identical requests."""
    res = {}

    def one(i, n_):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": CONC}],
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        j = json.load(urllib.request.urlopen(req, timeout=300))
        res[i] = (j.get("usage", {}).get("completion_tokens", 0), time.time() - t0)

    ths = [threading.Thread(target=one, args=(i, n)) for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    return sum(v[0] for v in res.values()) / max(wall, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18300")
    ap.add_argument("--model", default="nemotron-3.5-super-vl")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    chat = run(a.base, CHAT, 150, a.model)
    code = run(a.base, CODE, 200, a.model)
    c2 = conc(a.base, 2, a.model)
    c4 = conc(a.base, 4, a.model)

    out = {
        "label": a.label,
        "chat": round(chat, 1),
        "code": round(code, 1),
        "mean": round((chat + code) / 2, 1),
        "conc2": round(c2, 1),
        "conc4": round(c4, 1),
    }
    print(json.dumps(out))
    with open(f"result-{a.label}.json", "w") as f:
        f.write(json.dumps(out))


if __name__ == "__main__":
    main()