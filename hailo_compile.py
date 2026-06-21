"""Compile multideepc_encode.onnx -> multideepc_encode.hef using the Hailo
Dataflow Compiler (DFC). RUNS INSIDE the amd64 compile container only
(Dockerfile.compile), because the DFC is x86_64-Linux + Hailo-SDK only.

Pipeline: parse (ONNX -> HAR) -> optimize/quantize INT8 (needs calib) -> compile -> HEF.

Calibration: accuracy is out of scope for the deployment bring-up, so by default
this uses a SYNTHETIC calibration set (random, standard-normal - matches the
expected post-scaler input distribution). Swap `--calib` with a real assembled
[N,88] .npz once the scaler + CNN features are available, for production accuracy.

Target: hailo8l (Raspberry Pi AI Kit / AI HAT+, 13 TOPS). NOT hailo8.
"""
import argparse
import os
import numpy as np

IN_NAMES = ["x_ndvi", "x_env", "x_soil", "x_spat", "x_img"]
IN_DIMS = {"x_ndvi": 7, "x_env": 5, "x_soil": 7, "x_spat": 5, "x_img": 64}
HW_ARCH = "hailo8l"


def _raw_calib(n, seed=0, npz=None):
    """Per-ONNX-input arrays of shape [n, dim] (synthetic N(0,1) or from .npz)."""
    if npz:
        z = np.load(npz)
        return {name: z[name].astype(np.float32) for name in IN_NAMES}
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal((n, IN_DIMS[name])).astype(np.float32)
            for name in IN_NAMES}


def build_calib(runner, n, seed=0, npz=None):
    """Hailo keys calibration by the translated INPUT-LAYER name (e.g.
    'multideepc_encode/input_layer1'), not the ONNX tensor name, and expects
    NHWC [N,1,1,dim] tensors. Discover the mapping from the HN and reshape."""
    import json
    raw = _raw_calib(n, seed, npz)
    hn = runner.get_hn()
    d = json.loads(hn) if isinstance(hn, str) else hn
    calib = {}
    for hailo_name, L in d["layers"].items():
        if L.get("type") != "input_layer":
            continue
        onnx_name = L.get("original_names", [None])[0]
        if onnx_name not in raw:
            raise KeyError(f"input layer {hailo_name} maps to unknown {onnx_name}")
        shp = L["output_shapes"][0]          # [-1,1,1,dim]
        dim = shp[-1]
        calib[hailo_name] = raw[onnx_name].reshape(-1, *[s if s != -1 else 1
                                                         for s in shp[1:]]).astype(np.float32)
        assert dim == IN_DIMS[onnx_name], f"{onnx_name} dim {dim}!={IN_DIMS[onnx_name]}"
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/multideepc_encode.onnx")
    ap.add_argument("--out", default="artifacts/multideepc_encode.hef")
    ap.add_argument("--calib", default=None,
                    help="optional .npz with real [N,dim] arrays per input; "
                         "default = synthetic N(0,1)")
    ap.add_argument("--calib-n", type=int, default=256)
    args = ap.parse_args()

    # Imported here so the script can be inspected without the (gated) SDK present.
    from hailo_sdk_client import ClientRunner  # provided by the DFC wheel

    if not args.calib:
        print(f"[warn] using SYNTHETIC calibration (n={args.calib_n}). "
              "Swap --calib with real data for production accuracy.")

    runner = ClientRunner(hw_arch=HW_ARCH)

    print("[1/3] parse ONNX -> HAR")
    runner.translate_onnx_model(
        args.onnx, "multideepc_encode",
        start_node_names=IN_NAMES,
        net_input_shapes={n: [1, IN_DIMS[n]] for n in IN_NAMES},
    )

    print("[2/3] optimize / quantize INT8 (calibration)")
    calib = build_calib(runner, args.calib_n, npz=args.calib)
    runner.optimize(calib)

    print("[3/3] compile -> HEF")
    hef = runner.compile()
    with open(args.out, "wb") as f:
        f.write(hef)
    print("wrote:", args.out, f"({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
