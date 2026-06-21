"""Export the MultiDeepC ENCODE path (5 encoders + attention + fusion -> z[64])
to ONNX for Hailo compilation.

Deployment split (per design doc + Hailo constraints):
  - Hailo NPU : encode() = the heavy linear/bn/relu/sigmoid graph -> latent z[64]
  - Pi CPU    : DEC Student-t clustering head (tiny, verified alpha=1.0)

The CNN extractor (x_img) and feature scalers are UPSTREAM preprocessing, NOT
part of this graph. The model takes x_img[64] as an already-extracted input
tensor, so export does not need the CNN or the scaler. They are swapped in later.

Opset 11 (Hailo DFC compatible). BatchNorm runs in eval() -> folds into the
preceding Linear at compile time. Dynamic batch axis.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from multideepc import load_multideepc, _BRANCHES  # noqa: E402

CKPT = os.path.join(HERE, "models", "multideepc_modeA_model.pth")
OUT_DIR = os.path.join(HERE, "artifacts")
ONNX_PATH = os.path.join(OUT_DIR, "multideepc_encode.onnx")


class EncodeOnly(torch.nn.Module):
    """Wrapper exposing only encode() -> z, so the ONNX graph is exactly the
    Hailo-targeted subgraph."""
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x_ndvi, x_env, x_soil, x_spat, x_img):
        return self.m.encode(x_ndvi, x_env, x_soil, x_spat, x_img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, _ = load_multideepc(CKPT)
    wrap = EncodeOnly(model).eval()

    B = 4
    dims = {n: _BRANCHES[n][0] for n in ["ndvi", "env", "soil", "spat", "img"]}
    dummy = tuple(torch.randn(B, dims[n]) for n in ["ndvi", "env", "soil", "spat", "img"])
    in_names = ["x_ndvi", "x_env", "x_soil", "x_spat", "x_img"]

    torch.onnx.export(
        wrap, dummy, ONNX_PATH,
        input_names=in_names, output_names=["z"],
        opset_version=11,
        dynamic_axes={n: {0: "batch"} for n in in_names + ["z"]},
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter (no onnxscript dep)
    )
    print("exported:", ONNX_PATH, f"({os.path.getsize(ONNX_PATH)} bytes)")

    # validate: onnx checker + op inventory + numeric parity vs torch
    import onnx
    g = onnx.load(ONNX_PATH)
    onnx.checker.check_model(g)
    ops = sorted({n.op_type for n in g.graph.node})
    print("onnx.checker: PASS")
    print("opset:", g.opset_import[0].version, "| ir_version:", g.ir_version)
    print("ops used:", ops)

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    feeds = {n: d.numpy() for n, d in zip(in_names, dummy)}
    z_ort = sess.run(["z"], feeds)[0]
    with torch.no_grad():
        z_torch = wrap(*dummy).numpy()
    maxerr = np.abs(z_ort - z_torch).max()
    print(f"torch vs onnxruntime: max|dz|={maxerr:.3e} ->",
          "PASS" if maxerr < 1e-4 else "FAIL")

    # Hailo op-support sanity flag
    HAILO_OK = {"Gemm", "MatMul", "Add", "Mul", "Sigmoid", "Relu",
                "BatchNormalization", "Concat", "Constant", "Reshape", "Flatten"}
    unknown = [o for o in ops if o not in HAILO_OK]
    print("ops outside common-Hailo set:", unknown if unknown else "none")


if __name__ == "__main__":
    main()
