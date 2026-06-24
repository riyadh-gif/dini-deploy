"""PoC demo: kirim data grid ke layanan MultiDeepC di Raspberry Pi 5 + Hailo-8L,
tampilkan label klaster yang dikembalikan oleh NPU. Stdlib saja.

Pakai:
  python3 poc_demo.py                         # default http://localhost:8080 (jalankan di Pi)
  python3 poc_demo.py --url http://192.168.1.108:8080   # dari laptop ke Pi

Catatan: kompilasi memakai calibration sintetis, jadi LABEL belum bermakna untuk
keputusan lapangan. Yang didemokan: jalur deployment penuh (data -> NPU Hailo ->
label) berjalan di perangkat edge sebenarnya.
"""
import argparse
import json
import time
import urllib.request

# Empat contoh grid (nilai sensor dummy bentuknya benar: ndvi7/env5/soil7/spat5/img64).
# img di-nol-kan: ekstraktor CNN menyusul (lihat docs/scaler_status.md).
SAMPLES = [
    {"nama": "Grid A (sawah utara)",  "x_ndvi": [0.62, 0.58, 0.71, 0.10, 0.65, 0.0142, 0.55],
     "x_env": [31.0, 72.0, 470, 4, 6.6], "x_soil": [45, 18, 89, 0.14, 26.3, 41.4, 6.6],
     "x_spat": [113.789, -8.1545, 122, 0.0, 0.0]},
    {"nama": "Grid B (sawah timur)",  "x_ndvi": [0.21, 0.05, 0.46, 0.14, 0.18, 0.0206, 0.20],
     "x_env": [30.5, 78.0, 556, 6, 6.7], "x_soil": [49, 42, 60, 0.19, 27.0, 58.0, 6.7],
     "x_spat": [113.790, -8.1548, 112, 0.0, 0.0]},
    {"nama": "Grid C (petak kering)", "x_ndvi": [0.08, -0.2, 0.30, 0.12, 0.10, 0.030, 0.05],
     "x_env": [33.2, 65.0, 410, 3, 6.4], "x_soil": [20, 10, 40, 0.08, 28.0, 50.0, 6.3],
     "x_spat": [113.791, -8.1551, 95, 0.0, 0.0]},
]
for s in SAMPLES:
    s["x_img"] = [0.0] * 64


def post(url, body):
    req = urllib.request.Request(url + "/predict", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    args = ap.parse_args()

    print("=" * 64)
    print("  PoC: MultiDeepC di Raspberry Pi 5 + Hailo-8L (edge inference)")
    print("=" * 64)

    with urllib.request.urlopen(args.url + "/health", timeout=10) as r:
        h = json.load(r)
    print(f"\nLayanan   : {args.url}")
    print(f"Status    : {h['status']}")
    print(f"Backend   : {h['backend']}   (hailo = inferensi di NPU Hailo-8L)")
    print(f"Klaster   : {h['centers_shape'][0]} kelas\n")

    print(f"{'Grid':<24}{'Label':<14}{'Confidence':<12}{'Latensi':<10}{'Backend'}")
    print("-" * 64)
    for s in SAMPLES:
        body = {k: s[k] for k in ("x_ndvi", "x_env", "x_soil", "x_spat", "x_img")}
        t0 = time.perf_counter()
        res = post(args.url, body)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{s['nama']:<24}{res['label']:<14}{res['confidence']:<12.3f}"
              f"{ms:<10.1f}{res['backend']}")

    print("-" * 64)
    print("\nKesimpulan: data sensor -> NPU Hailo-8L -> label klaster, berjalan")
    print("di perangkat edge. (Akurasi menyusul scaler + CNN; lihat docs.)")


if __name__ == "__main__":
    main()
