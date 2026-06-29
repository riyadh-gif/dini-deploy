"""Inference-time benchmark on Pi5 + Hailo-8L, using real grid inputs.
Reports NPU encode latency percentiles + full-path (encode+DEC head) + per-grid labels.
Inference time is independent of input values; real grids used for honest demo."""
import json, time, statistics
import numpy as np
import dec_head
from hailo_platform import (VDevice, HEF, ConfigureParams, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType)

HEF_PATH = "artifacts/multideepc_encode.hef"
L2B = {"input_layer1": "x_ndvi", "input_layer2": "x_env", "input_layer3": "x_soil",
       "input_layer4": "x_spat", "input_layer5": "x_img"}

grids = json.load(open("real_grids.json"))
centers = dec_head.load_cluster_centers()
hef = HEF(HEF_PATH)
in_infos = hef.get_input_vstream_infos()
out_name = hef.get_output_vstream_infos()[0].name

def feeds_for(g):
    return {info.name: np.asarray(g[L2B[info.name.split("/")[-1]]], dtype=np.float32)
            .reshape((1,) + tuple(info.shape)) for info in in_infos}

def stats(a):
    a = sorted(a); n = len(a)
    return min(a), statistics.mean(a), statistics.median(a), a[int(n*0.95)], a[int(n*0.99)], max(a)

with VDevice() as dev:
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = dev.configure(hef, cfg)[0]
    ngp = ng.create_params()
    inp = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    outp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    f0 = feeds_for(grids[0])
    with InferVStreams(ng, inp, outp) as pipe:
        with ng.activate(ngp):
            for _ in range(50):
                pipe.infer(f0)                      # warmup
            N = 2000; ts = []
            for _ in range(N):
                t = time.perf_counter(); pipe.infer(f0); ts.append((time.perf_counter()-t)*1000)
            tf = []
            for _ in range(500):
                t = time.perf_counter()
                out = pipe.infer(f0); z = np.asarray(out[out_name]).reshape(1, 64)
                dec_head.predict(z, centers)
                tf.append((time.perf_counter()-t)*1000)
            rows = []
            for g in grids:
                out = pipe.infer(feeds_for(g)); z = np.asarray(out[out_name]).reshape(1, 64)
                rows.append((g["nama"], dec_head.predict(z, centers)[0]))

mn, me, md, p95, p99, mx = stats(ts)
print("=" * 60)
print(f"INFERENCE TIME - NPU encode only  (N={N}, input grid asli)")
print("=" * 60)
print(f"  min {mn:.3f} | mean {me:.3f} | median {md:.3f} | p95 {p95:.3f} | p99 {p99:.3f} | max {mx:.3f}  ms")
print(f"  throughput ~ {1000/me:,.0f} inferensi/detik")
mn2, me2, md2, *_ = stats(tf)
print(f"\nFull path (encode NPU + DEC head CPU, N=500): mean {me2:.3f} | median {md2:.3f} ms")
print("\nReal grids -> label:")
for nama, p in rows:
    print(f"  {nama:<32} {p['label']:<12} conf {p['confidence']:.3f}")
