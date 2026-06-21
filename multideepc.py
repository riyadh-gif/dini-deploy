"""
MultiDeepC architecture, reconstructed from the trained checkpoint
output_multideepc_modeA/multideepc_modeA_model.pth (state_dict only - no
original source code existed in the project).

Reconstruction is driven by the EXACT tensor names/shapes in the state_dict
(115 tensors). Every parametric/buffer layer (Linear, BatchNorm1d, cluster
centers) is placed so that `load_state_dict(..., strict=True)` succeeds.

VERIFIED numerically (see verify_multideepc.py):
  - DEC clustering head: Student-t soft assignment, alpha=1.0, reproduces
    Q_soft.npy from Z_latent.npy + cluster_centers to float precision
    (max abs err ~4.8e-7).

ASSUMED (cannot be numerically verified - the assembled input matrix X that
produced Z_latent.npy was not saved, and the img branch needs the missing
CNN extractor output_cnn_nirred/):
  - Hidden activations are ReLU.
  - Non-parametric slots between linears are ReLU then Dropout (encoder/fusion)
    or a single ReLU (decoder/attention). Dropout is a no-op in eval() so it
    does not change inference numerics regardless of its exact rate.
  - Attention applies an element-wise sigmoid gate: out = z * sigmoid(MLP(z)).
These assumptions affect ONLY the encoder/fusion path (the part destined for
Hailo). They must be re-confirmed once X or the CNN extractor is recovered.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Branch input dims and encoder layer widths, read directly from the state_dict.
# name: (in_dim, hidden, bottleneck)
_BRANCHES = {
    "ndvi": (7, 64, 32),
    "env":  (5, 64, 32),
    "soil": (7, 64, 32),
    "spat": (5, 32, 16),
    "img":  (64, 128, 64),
}
FUSION_IN = 32 + 32 + 32 + 16 + 64  # = 176, matches fusion.fc.0.weight (128,176)
LATENT_DIM = 64
N_CLUSTERS = 4


class AEBranch(nn.Module):
    """Per-modality autoencoder. Only `encode()` is used at inference; the
    decoder exists so strict state_dict loading succeeds (it was used in the
    pretraining stage)."""

    def __init__(self, in_dim, hidden, bottleneck, dropout=0.0):
        super().__init__()
        # indices: 0 Linear, 1 BN, 2 ReLU, 3 Dropout, 4 Linear, 5 BN
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.BatchNorm1d(bottleneck),
        )
        # indices: 0 Linear, 1 ReLU, 2 Linear
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class Attention(nn.Module):
    """Element-wise gating attention on a branch latent.
    attn = Sequential(Linear(d,d), ReLU, Linear(d,d)); gate = sigmoid(attn(z))."""

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z):
        return z * torch.sigmoid(self.attn(z))


class Fusion(nn.Module):
    """Concatenated branch latents (176) -> latent z (64)."""

    def __init__(self, in_dim=FUSION_IN, hidden=128, out=LATENT_DIM, dropout=0.0):
        super().__init__()
        # indices: 0 Linear, 1 ReLU, 2 Dropout, 3 Linear
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.fc(x)


class ClusteringHead(nn.Module):
    """DEC Student-t soft assignment. VERIFIED: alpha=1.0."""

    def __init__(self, n_clusters=N_CLUSTERS, latent_dim=LATENT_DIM, alpha=1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.alpha = alpha

    def forward(self, z):
        d = torch.sum((z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        num = (1.0 + d / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        return num / torch.sum(num, dim=1, keepdim=True)


class MultiDeepC(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.enc_ndvi = AEBranch(*_BRANCHES["ndvi"])
        self.enc_env  = AEBranch(*_BRANCHES["env"])
        self.enc_soil = AEBranch(*_BRANCHES["soil"])
        self.enc_spat = AEBranch(*_BRANCHES["spat"])
        self.enc_img  = AEBranch(*_BRANCHES["img"])

        self.attn_ndvi = Attention(_BRANCHES["ndvi"][2])
        self.attn_env  = Attention(_BRANCHES["env"][2])
        self.attn_soil = Attention(_BRANCHES["soil"][2])
        self.attn_spat = Attention(_BRANCHES["spat"][2])
        self.attn_img  = Attention(_BRANCHES["img"][2])

        self.fusion  = Fusion()
        self.cluster = ClusteringHead(alpha=alpha)

    def encode(self, x_ndvi, x_env, x_soil, x_spat, x_img):
        """The Hailo-targeted path: 5 encoders -> attention -> fusion -> z(64)."""
        z_ndvi = self.attn_ndvi(self.enc_ndvi.encode(x_ndvi))
        z_env  = self.attn_env(self.enc_env.encode(x_env))
        z_soil = self.attn_soil(self.enc_soil.encode(x_soil))
        z_spat = self.attn_spat(self.enc_spat.encode(x_spat))
        z_img  = self.attn_img(self.enc_img.encode(x_img))
        cat = torch.cat([z_ndvi, z_env, z_soil, z_spat, z_img], dim=1)
        return self.fusion(cat)

    def soft_assign(self, z):
        """The CPU-targeted DEC head."""
        return self.cluster(z)

    def forward(self, x_ndvi, x_env, x_soil, x_spat, x_img):
        z = self.encode(x_ndvi, x_env, x_soil, x_spat, x_img)
        return z, self.soft_assign(z)


def load_multideepc(ckpt_path, map_location="cpu"):
    """Build the module and strict-load the trained weights. Returns (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model = MultiDeepC(alpha=1.0)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt
