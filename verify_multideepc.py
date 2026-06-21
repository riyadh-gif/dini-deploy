"""Step-1 verification harness for the reconstructed MultiDeepC module.

Checks that are FULLY decisive (pass/fail):
  1. strict load_state_dict  -> architecture names+shapes match all 115 tensors
  2. clustering head numerics -> reproduce Q_soft.npy from Z_latent.npy + centers
  3. forward smoke test       -> encode() + soft_assign() run, correct output shapes

Cannot be verified here (documented gap): encoder/fusion numeric output, because
the assembled input matrix X that produced Z_latent.npy was not saved and the img
branch needs the missing CNN extractor (output_cnn_nirred/).
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from multideepc import load_multideepc, _BRANCHES  # noqa: E402

CKPT = os.path.join(HERE, "models", "multideepc_modeA_model.pth")
OUT = os.path.join(HERE, "reference")


def main():
    ok = True

    # 1. strict load
    model, ckpt = load_multideepc(CKPT)
    print("[1] strict load_state_dict: PASS (all 115 tensors matched, model.eval())")

    # 2. clustering head numerics through the module
    Z = np.load(os.path.join(OUT, "Z_latent.npy")).astype(np.float32)
    Q = np.load(os.path.join(OUT, "Q_soft.npy")).astype(np.float32)
    yp = np.load(os.path.join(OUT, "y_pred.npy"))
    with torch.no_grad():
        Qh = model.soft_assign(torch.from_numpy(Z)).numpy()
    maxerr = np.abs(Qh - Q).max()
    argmatch = (Qh.argmax(1) == Q.argmax(1)).mean()
    ymatch = (Qh.argmax(1) == yp).mean()
    print(f"[2] clustering head: max|q_hat-q|={maxerr:.3e}  "
          f"argmax==Q:{argmatch:.4f}  argmax==y_pred:{ymatch:.4f}")
    if not (maxerr < 1e-4 and argmatch == 1.0 and ymatch == 1.0):
        ok = False
        print("    -> FAIL")
    else:
        print("    -> PASS (DEC head reproduces saved outputs exactly)")

    # 3. forward smoke test on dummy input (eval mode -> BatchNorm uses running stats)
    B = 8
    xs = {n: torch.randn(B, d[0]) for n, d in _BRANCHES.items()}
    with torch.no_grad():
        z, q = model(xs["ndvi"], xs["env"], xs["soil"], xs["spat"], xs["img"])
    shape_ok = tuple(z.shape) == (B, 64) and tuple(q.shape) == (B, 4)
    qsum_ok = bool(torch.allclose(q.sum(1), torch.ones(B), atol=1e-5))
    print(f"[3] forward smoke: z{tuple(z.shape)} q{tuple(q.shape)} "
          f"rows_sum_to_1={qsum_ok}")
    if not (shape_ok and qsum_ok):
        ok = False
        print("    -> FAIL")
    else:
        print("    -> PASS (encode->fusion->head runs, shapes + simplex OK)")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
