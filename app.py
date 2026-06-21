"""On-device inference server for MultiDeepC on Raspberry Pi 5 + Hailo 8L.

Flow per request:
  raw features (5 branches) -> scaler.transform (CPU) -> encode (NPU or CPU)
  -> latent z[64] -> DEC head (CPU) -> cluster label.

Two backends, selected by env BACKEND:
  - hailo : run the compiled .hef on the Hailo-8L via HailoRT  (production, on Pi)
  - onnx  : run multideepc_encode.onnx via onnxruntime on CPU  (no NPU needed)

The `onnx` backend lets the hardware team validate the whole container/plumbing
on ANY machine before the Hailo HEF + driver are wired up. Same REST contract.

Placeholders (swap when artifacts arrive - see STEP2_scaler_status.md):
  - scaler  : IdentityScaler (no-op). Replace with real per-branch StandardScaler.
  - x_img   : accepted as a 64-vector input. Upstream CNN extractor produces it;
              until then callers may send zeros. Model deploy does NOT need the CNN.
"""
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from dec_head import load_cluster_centers, predict, IdentityScaler, BRANCH_DIMS

BACKEND = os.environ.get("BACKEND", "onnx").lower()
ONNX_PATH = os.environ.get("ONNX_PATH", "artifacts/multideepc_encode.onnx")
HEF_PATH = os.environ.get("HEF_PATH", "artifacts/multideepc_encode.hef")
IN_NAMES = ["x_ndvi", "x_env", "x_soil", "x_spat", "x_img"]

app = FastAPI(title="MultiDeepC Pi5+Hailo8L inference", version="0.1")
_centers = load_cluster_centers()
_scaler = IdentityScaler()  # PLACEHOLDER
_encoder = None


class GridInput(BaseModel):
    x_ndvi: list[float] = Field(..., min_length=7, max_length=7)
    x_env:  list[float] = Field(..., min_length=5, max_length=5)
    x_soil: list[float] = Field(..., min_length=7, max_length=7)
    x_spat: list[float] = Field(..., min_length=5, max_length=5)
    x_img:  list[float] = Field(..., min_length=64, max_length=64)


class OnnxEncoder:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def encode(self, feats):  # feats: dict name->[1,dim] float32
        return self.sess.run(["z"], feats)[0]


class HailoEncoder:
    """Minimal HailoRT wrapper. Exact API differs by HailoRT version on the Pi;
    confirm against the installed hailort before production."""
    def __init__(self, hef_path):
        from hailo_platform import (VDevice, HEF, ConfigureParams,
                                    HailoStreamInterface)
        self.VDevice = VDevice
        self.hef = HEF(hef_path)
        self.dev = VDevice()
        cfg = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.dev.configure(self.hef, cfg)[0]

    def encode(self, feats):
        from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams
        ivp = InputVStreamParams.make(self.network_group)
        ovp = OutputVStreamParams.make(self.network_group)
        with InferVStreams(self.network_group, ivp, ovp) as pipe:
            with self.network_group.activate():
                out = pipe.infer(feats)
        return np.asarray(list(out.values())[0], dtype=np.float32)


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = HailoEncoder(HEF_PATH) if BACKEND == "hailo" else OnnxEncoder(ONNX_PATH)
    return _encoder


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND,
            "scaler": "identity-PLACEHOLDER", "centers_shape": list(_centers.shape)}


@app.post("/predict")
def do_predict(grid: GridInput):
    feats = {}
    for name in IN_NAMES:
        key = name.split("_", 1)[1]  # ndvi/env/...
        v = _scaler.transform(getattr(grid, name))
        feats[name] = np.asarray(v, dtype=np.float32).reshape(1, BRANCH_DIMS[key])
    z = get_encoder().encode(feats)
    result = predict(z, _centers)[0]
    result["backend"] = BACKEND
    return result
