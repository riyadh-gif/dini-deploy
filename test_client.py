#!/usr/bin/env python3
"""Standalone smoke-test client for the MultiDeepC inference server.

No third-party deps -- stdlib (urllib + json + random) only, so it runs on the
Pi, on the dev box, or anywhere with plain python3.

It hits two endpoints of app.py:
  GET  /health   -> prints backend / scaler / centers_shape
  POST /predict  -> sends one sample grid, prints the cluster result JSON

Input branch shapes (must match app.py GridInput exactly):
  x_ndvi[7]  x_env[5]  x_soil[7]  x_spat[5]  x_img[64]

Usage:
  python test_client.py                       # full random-ish sample
  python test_client.py --zeros               # x_img = zeros (CNN-missing case)
  python test_client.py --url http://PI_IP:8080
  python test_client.py --seed 123

The --zeros flag models the real deploy bring-up state: the upstream CNN that
produces x_img[64] is MISSING, so callers send zeros for x_img. The 4 sensor
branches still get small random values. Accuracy is OUT OF SCOPE here -- this
only verifies the container/REST plumbing end to end.
"""
import argparse
import json
import random
import urllib.error
import urllib.request

# Branch dimensions -- mirror app.py GridInput / dec_head.BRANCH_DIMS.
BRANCH_DIMS = {"x_ndvi": 7, "x_env": 5, "x_soil": 7, "x_spat": 5, "x_img": 64}


def small_random(n, rng):
    """n small-ish floats; realistic post-scaler-ish ~N(0,1) magnitudes."""
    return [round(rng.gauss(0.0, 1.0), 6) for _ in range(n)]


def build_sample(zeros_img=False, seed=0):
    rng = random.Random(seed)
    sample = {}
    for name, dim in BRANCH_DIMS.items():
        if name == "x_img" and zeros_img:
            sample[name] = [0.0] * dim          # CNN-missing case
        else:
            sample[name] = small_random(dim, rng)
    return sample


def _get(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="MultiDeepC inference smoke test")
    ap.add_argument("--url", default="http://localhost:8080",
                    help="base URL of the server (default http://localhost:8080)")
    ap.add_argument("--zeros", action="store_true",
                    help="send zeros for x_img (the CNN-missing deploy case)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for the sensor-branch values")
    args = ap.parse_args()

    base = args.url.rstrip("/")

    # 1) /health
    print(f"GET  {base}/health")
    try:
        health = _get(base + "/health")
        print(json.dumps(health, indent=2))
    except urllib.error.URLError as e:
        print("  [ERROR] could not reach /health:", e)
        print("  Is the server running?  uvicorn app:app --host 0.0.0.0 --port 8080")
        return 1

    # 2) /predict
    sample = build_sample(zeros_img=args.zeros, seed=args.seed)
    print(f"\nPOST {base}/predict   (x_img={'zeros' if args.zeros else 'random'}, seed={args.seed})")
    print("  branch shapes:", {k: len(v) for k, v in sample.items()})
    try:
        result = _post_json(base + "/predict", sample)
        print(json.dumps(result, indent=2))
    except urllib.error.HTTPError as e:
        print("  [ERROR] HTTP", e.code, e.reason)
        print("  body:", e.read().decode("utf-8", "replace"))
        return 1
    except urllib.error.URLError as e:
        print("  [ERROR] could not reach /predict:", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
