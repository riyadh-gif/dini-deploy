"""CPU-side DEC clustering head + input preprocessing for on-device inference.

This runs on the Raspberry Pi 5 CPU. It takes the latent z[64] produced by the
Hailo NPU (the compiled encode graph) and returns the cluster label.

VERIFIED: Student-t soft assignment, alpha=1.0, reproduces the trained model's
Q_soft.npy to float precision.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LABELS = {0: "Non-Tanaman", 1: "Kritis", 2: "Sedang", 3: "Sehat"}

# Branch feature order/selection - must match training. UNKNOWN until the student
# supplies the assembly script; this order is a PLACEHOLDER for pipeline bring-up.
BRANCH_DIMS = {"ndvi": 7, "env": 5, "soil": 7, "spat": 5, "img": 64}


def load_cluster_centers(ckpt_path=None):
    """Prefer the pre-extracted artifacts/cluster_centers.npy (torch-free, used
    in the arm64 runtime container). Fall back to reading the .pth via torch."""
    npy = os.path.join(HERE, "artifacts", "cluster_centers.npy")
    if os.path.exists(npy):
        return np.load(npy).astype(np.float32)  # (4,64)
    import torch
    ckpt_path = ckpt_path or os.path.join(
        HERE, "models", "multideepc_modeA_model.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return np.asarray(ckpt["cluster_centers"], dtype=np.float32)  # (4,64)


def soft_assign(z, centers, alpha=1.0):
    z = np.atleast_2d(z).astype(np.float32)
    d = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    num = (1.0 + d / alpha) ** (-(alpha + 1.0) / 2.0)
    return num / num.sum(1, keepdims=True)


def predict(z, centers, alpha=1.0):
    q = soft_assign(z, centers, alpha)
    idx = q.argmax(1)
    return [{"label_id": int(i), "label": LABELS[int(i)],
             "confidence": float(q[k, i])} for k, i in enumerate(idx)]


class IdentityScaler:
    """PLACEHOLDER normalizer (no-op). Pipeline runs, numbers are NOT correct.
    Replace with the real per-branch StandardScaler (mean_/scale_) from training."""
    def transform(self, x):
        return np.asarray(x, dtype=np.float32)


if __name__ == "__main__":
    # self-check against saved outputs
    Z = np.load(os.path.join(HERE, "reference", "Z_latent.npy"))
    Q = np.load(os.path.join(HERE, "reference", "Q_soft.npy"))
    c = load_cluster_centers()
    q = soft_assign(Z, c)
    print("DEC head self-check max|dq| =", np.abs(q - Q).max())
