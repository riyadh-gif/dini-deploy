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


# HEF input-layer name (suffix) -> our branch input name. Fixed at compile time
# (translate_onnx_model original_names: layer1=ndvi, 2=env, 3=soil, 4=spat, 5=img).
_LAYER2BRANCH = {
    "input_layer1": "x_ndvi", "input_layer2": "x_env", "input_layer3": "x_soil",
    "input_layer4": "x_spat", "input_layer5": "x_img",
}


class HailoEncoder:
    """Runs the compiled HEF on the Hailo-8L via HailoRT (verified against
    HailoRT 4.23.0). FLOAT32 in/out so HailoRT auto-quantizes around the UINT8
    net; feeds are keyed by the HEF vstream names, mapped from our branch names."""
    def __init__(self, hef_path):
        from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, FormatType)
        self.hef = HEF(hef_path)
        self.dev = VDevice()
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.dev.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()
        self.in_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.out_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.in_infos = self.hef.get_input_vstream_infos()
        self.out_name = self.hef.get_output_vstream_infos()[0].name

    def encode(self, feats):  # feats: {x_ndvi:[1,7], ...} float32
        from hailo_platform import InferVStreams
        feeds = {}
        for info in self.in_infos:
            branch = _LAYER2BRANCH[info.name.split("/")[-1]]
            feeds[info.name] = feats[branch].reshape((1,) + tuple(info.shape)).astype(np.float32)
        with InferVStreams(self.network_group, self.in_params, self.out_params) as pipe:
            with self.network_group.activate(self.ng_params):
                out = pipe.infer(feeds)
        return np.asarray(out[self.out_name], dtype=np.float32).reshape(1, 64)


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
