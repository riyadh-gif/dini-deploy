# Dini Deploy

Deployment pipeline untuk model **MultiDeepC** (clustering kondisi lahan padi per grid)
ke perangkat tepi **Raspberry Pi 5 + Hailo-8L AI accelerator (13 TOPS)**.

Repositori ini berisi seluruh kode, artefak, dan dokumentasi yang diperlukan untuk
mengubah model PyTorch hasil pelatihan menjadi format yang dapat dijalankan di
akselerator Hailo, beserta server inferensi dan paket container untuk tim hardware.

## Ringkasan Status

| Tahap | Keterangan | Status |
|-------|-----------|--------|
| 1. Rekonstruksi model | Bangun ulang arsitektur dari checkpoint, muat bobot | Selesai, terverifikasi |
| 2. Ekspor ONNX | Konversi jalur encoder ke ONNX opset 11 | Selesai |
| 3. Server inferensi | FastAPI dengan dua backend (CPU dan Hailo) | Selesai, teruji |
| 4. Kompilasi HEF | ONNX ke format Hailo (.hef) via Dataflow Compiler | Selesai |
| 5. Paket container | Image runtime arm64 (CPU dan Hailo) dan image compile amd64 | Selesai |
| 6. Verifikasi di perangkat | Uji muat dan inferensi HEF di Raspberry Pi sebenarnya | Selesai, teruji di hardware |

Inti pekerjaan deployment telah selesai dan terverifikasi di hardware. Pipeline
penuh dari `.pth` sampai inferensi di NPU sudah terbukti berjalan di Raspberry Pi 5
dengan Hailo-8L, baik secara native maupun di dalam container Docker.

## Verifikasi di Hardware (Raspberry Pi 5 + Hailo-8L)

```
HailoRT pada Pi    : 4.23.0 (firmware + runtime + driver)
Device             : HAILO8L, /dev/hailo0 aktif
HEF dikompilasi    : Dataflow Compiler 3.33.1 (forward-compatible, load tanpa error)
Throughput NPU     : ~15.480 FPS (hailortcli run)
Inferensi native   : encode NPU -> DEC head -> label, latensi ~0.34 ms/inferensi
Inferensi container: image multideepc-runtime:hailo, /predict mengembalikan label
                     melalui NPU (BACKEND=hailo, --device /dev/hailo0)
```

Catatan: nilai prediksi belum final karena kompilasi memakai calibration sintetis.
Akurasi produksi memerlukan scaler pelatihan dan ekstraktor CNN, lalu kompilasi ulang.
Yang terbukti di sini adalah jalur deployment penuh berjalan di hardware.

## Arsitektur Deployment

Model dipecah menjadi dua bagian sesuai karakteristik komputasi dan dukungan Hailo:

```
  Raspberry Pi 5 (CPU)                         Hailo-8L (NPU)
  +-----------------------+                    +------------------------+
  | Preprocessing/scaler  |  fitur 5 cabang    | Encoder (5 cabang MLP) |
  | DEC clustering head   | -----------------> | Attention fusion       |
  | (Student-t, alpha=1)  | <----- z[64] ----- | Output latent z[64]    |
  +-----------------------+                    +------------------------+
            |
            v
     Label cluster (0..3)
```

Bagian berat (encoder dan fusion) dijalankan di NPU Hailo sebagai file `.hef`.
Bagian ringan dan non-standar (DEC clustering head) tetap di CPU. Pembagian ini
mengikuti dokumen desain dan batasan operator yang didukung Hailo.

### Spesifikasi Model

- Input: 5 cabang modalitas, total 88 fitur
  (ndvi 7, env 5, soil 7, spat 5, img 64).
- Output: latent `z[64]`, lalu soft assignment ke 4 cluster.
- Label: `0 = Non-Tanaman`, `1 = Kritis`, `2 = Sedang`, `3 = Sehat`.
- Clustering head: Student-t (DEC), `alpha = 1.0`. Telah diverifikasi cocok dengan
  output pelatihan hingga presisi float (selisih maksimum 5.4e-7).

## Hasil Kompilasi Hailo

```
Target arsitektur : hailo8l (Hailo-8L, 13 TOPS)
Dataflow Compiler : versi 3.33.1
Operator ONNX     : Gemm, BatchNormalization, Relu, Sigmoid, Mul, Concat
                    (seluruhnya didukung Hailo, tidak ada operator yang ditolak)
Pemetaan          : Successful Mapping (4 cluster, utilisasi 60.9%)
Kompilasi         : Successful Compilation
Ukuran HEF        : 762 KB
```

## Struktur Repositori

```
Dini deploy/
  multideepc.py            Definisi arsitektur MultiDeepC (rekonstruksi)
  dec_head.py              DEC clustering head untuk CPU + preprocessing
  export_onnx.py           Ekspor jalur encoder ke ONNX
  hailo_compile.py         Kompilasi ONNX ke HEF via Hailo Dataflow Compiler
  verify_multideepc.py     Harness verifikasi (muat bobot, cek numerik)
  app.py                   Server inferensi FastAPI (backend onnx atau hailo)
  test_client.py           Klien uji untuk endpoint /predict

  artifacts/
    multideepc_encode.onnx Graf encoder dalam ONNX
    multideepc_encode.hef  Model Hailo terkompilasi (siap untuk Pi)
    cluster_centers.npy    Pusat cluster untuk DEC head (tanpa dependensi torch)

  models/
    multideepc_modeA_model.pth   Checkpoint PyTorch hasil pelatihan

  reference/
    Z_latent.npy, Q_soft.npy, y_pred.npy   Output pelatihan untuk verifikasi

  docker/
    Dockerfile.hailo       Image arm64 yang memakai NPU Hailo (produksi di Pi)
    Dockerfile.runtime     Image arm64 backend CPU/onnx (bring-up, lintas mesin)
    Dockerfile.compile     Image amd64 untuk kompilasi HEF
    docker-compose.yml     Orkestrasi layanan runtime
    .dockerignore

  docs/
    HANDOFF.md             Panduan serah terima untuk tim hardware
    scaler_status.md       Status preprocessing dan langkah selanjutnya

  requirements-runtime.txt Dependensi server inferensi
  requirements-compile.txt Dependensi sisi kompilasi
```

## Cara Menjalankan

### 1. Verifikasi model (tanpa perangkat khusus)

```bash
pip install torch numpy onnx onnxruntime
python verify_multideepc.py
```

Memuat checkpoint, melakukan strict load atas 115 tensor, dan memeriksa bahwa
clustering head mereproduksi output pelatihan.

### 2. Ekspor ulang ONNX (opsional)

```bash
python export_onnx.py
```

Menghasilkan `artifacts/multideepc_encode.onnx` dan memvalidasi kesetaraan numerik
antara PyTorch dan ONNX Runtime.

### 3. Kompilasi HEF (di mesin Linux x86_64 dengan Hailo Dataflow Compiler)

```bash
python hailo_compile.py --onnx artifacts/multideepc_encode.onnx \
                        --out artifacts/multideepc_encode.hef
```

Dataflow Compiler hanya berjalan di Linux x86_64 dan memerlukan akun Hailo
Developer Zone untuk mengunduh paketnya.

### 4. Server inferensi (mode CPU, dapat diuji di mesin mana pun)

```bash
pip install -r requirements-runtime.txt
BACKEND=onnx uvicorn app:app --host 0.0.0.0 --port 8080
python test_client.py
```

### 5. Container Hailo di Raspberry Pi 5 (memakai NPU)

Dijalankan di Pi yang sudah memiliki `hailo-all` dan device `/dev/hailo0`.
Image dibangun native di Pi (arm64). Bridge NAT Docker pada sebagian Pi bermasalah,
jadi build memakai host networking.

```bash
# build di Pi (dari root repo)
sudo docker build --network=host -f docker/Dockerfile.hailo -t multideepc-runtime:hailo .

# jalankan dengan NPU di-passthrough
sudo docker run -d --name mdc-hailo --device /dev/hailo0 -p 8080:8080 \
     -e BACKEND=hailo multideepc-runtime:hailo

# uji
curl -s http://localhost:8080/health
python3 test_client.py
```

Contoh respons `/predict` (via NPU): `{"label_id":1,"label":"Kritis","backend":"hailo"}`.

## Catatan Penting

1. **Akurasi belum final.** Kompilasi saat ini memakai calibration sintetis untuk
   membuktikan pipeline. Untuk akurasi produksi, diperlukan scaler pelatihan dan
   ekstraktor fitur citra (CNN), lalu kompilasi ulang. Detail di `docs/scaler_status.md`.

2. **Pencocokan versi (sudah terverifikasi).** File HEF dikompilasi dengan Dataflow
   Compiler 3.33.1 dan terbukti dimuat serta berjalan di HailoRT 4.23.0 pada Pi
   tanpa error (forward-compatible). Untuk Pi dengan HailoRT lebih lama, periksa
   dengan `hailortcli fw-control identify` dan kompilasi ulang dengan versi
   compiler yang sesuai (prosesnya cepat).

3. **Pembaruan ke depan.** Saat ekstraktor CNN dan scaler tersedia, jalur citra
   dapat diaktifkan penuh tanpa mengubah arsitektur container.

## Lisensi

Internal, untuk keperluan penelitian.
