# README_HANDOFF - MultiDeepC di Raspberry Pi 5 + Hailo-8L

Dokumen serah-terima untuk **tim hardware**. Tujuan tahap ini (*bring-up*):
menghasilkan container yang bisa dijalankan tim hardware, BUKAN mengejar akurasi.
Perintah, path, dan istilah teknis sengaja dibiarkan dalam bahasa Inggris.

---

## (a) Gambaran arsitektur - dua bagian

Model MultiDeepC dipecah menjadi **dua bagian** saat deploy. Bagian **encode**
(5 cabang MLP kecil -> attention dengan *sigmoid gate* -> fusion -> latent
`z[64]`) dikompilasi menjadi file `.hef` dan berjalan di **Hailo-8L NPU**.
Bagian **DEC clustering head** (Student-t soft assignment, `alpha=1.0`, sudah
**VERIFIED**) berjalan di **CPU Raspberry Pi**: ia menerima `z[64]` dari NPU dan
mengeluarkan label cluster (0=Non-Tanaman, 1=Kritis, 2=Sedang, 3=Sehat). Server
FastAPI (`app.py`) menyatukan keduanya di balik endpoint REST `/health` dan
`/predict`, dan bisa memakai backend `onnx` (CPU saja, untuk bring-up) atau
`hailo` (NPU, produksi).

---

## (b) Daftar artifacts

Semua ada di folder `artifacts/` (kecuali `.hef` yang belum jadi):

| File | Isi |
|------|-----|
| `artifacts/multideepc_encode.onnx` | Subgraph **encode** dalam ONNX (opset 11, ir_version 6). Dipakai backend `onnx` (CPU) DAN sebagai input kompilasi Hailo. Sudah diverifikasi parity torch<->onnxruntime (~7e-7). |
| `artifacts/multideepc_encode.hef` | Hasil kompilasi Hailo (**BELUM ADA** - dibuat oleh `hailo_compile.py` / `Dockerfile.compile`, lihat bagian d). Ini yang dijalankan di NPU. |
| `artifacts/cluster_centers.npy` | Pusat cluster `(4, 64)` untuk DEC head di CPU. |
| `app.py` | Server FastAPI (`/health`, `/predict`). |
| `dec_head.py` | DEC head CPU + `IdentityScaler` (PLACEHOLDER) + label. |
| `test_client.py` | Skrip smoke-test stdlib-only (lihat bagian e). |
| `Dockerfile.runtime` | Image runtime arm64 untuk Pi. |
| `Dockerfile.compile` | Image kompilasi Hailo DFC (amd64 saja). |

---

## (c) PERINGATAN PLACEHOLDER (wajib dibaca)

Tahap ini **bring-up plumbing**, BUKAN produksi yang akurat:

1. **Scaler = IdentityScaler (no-op).** `StandardScaler` per-cabang dari training
   belum tersedia, jadi input TIDAK dinormalisasi. Angka prediksi belum benar.
   *Swap-in:* ganti `IdentityScaler` di `app.py`/`dec_head.py` dengan scaler asli
   (`mean_`/`scale_`) begitu file scaler tersedia.
2. **CNN extractor untuk `x_img` HILANG.** Fitur `x_img[64]` mestinya berasal
   dari CNN terpisah (`output_cnn_nirred/`) yang belum ada. Untuk bring-up,
   **caller boleh mengirim zeros untuk `x_img`** (lihat flag `--zeros` di
   `test_client.py`). *Swap-in:* sambungkan output CNN ke field `x_img` saat CNN
   sudah jadi.
3. **Akurasi OUT OF SCOPE.** Kalibrasi kompilasi Hailo memakai data sintetis
   `N(0,1)`. Target tahap ini = container yang jalan & REST contract yang benar.
   Akurasi diperbaiki belakangan setelah scaler + CNN tiba (ganti `--calib`
   dengan data `[N,dim]` asli saat re-compile).

---

## (d) BUILD image runtime arm64 di mesin x86 (buildx)

Image runtime untuk Pi (arm64) dibuild di mesin dev x86 memakai `buildx` + QEMU.
Jalankan dari folder `deploy/`:

```bash
# sekali saja: aktifkan emulasi arm64
docker run --privileged --rm tonistiigi/binfmt --install arm64

# build image arm64 (backend default = onnx)
docker buildx build --platform linux/arm64 \
    -f Dockerfile.runtime \
    -t multideepc-runtime:arm64 \
    --load .

# (opsional) ekspor ke tar untuk dipindah ke Pi tanpa registry
docker save multideepc-runtime:arm64 -o multideepc-runtime-arm64.tar
```

> Catatan: `.hef` BELUM disalin ke dalam image (baris `COPY ... .hef` di
> `Dockerfile.runtime` masih dikomentari). Untuk backend `hailo`, taruh `.hef`
> ke `artifacts/` dan aktifkan baris itu, atau mount `.hef` saat `docker run`.

### Membuat `.hef` (kompilasi Hailo, di mesin **amd64**)

Kompilasi WAJIB di host x86_64 (DFC tidak ada untuk arm64). Wheel DFC **gated**
- download manual dari Hailo Developer Zone (tidak ada di PyPI publik):

```bash
# dari folder deploy/, di host amd64, .whl sudah ada di context
docker build -f Dockerfile.compile \
    --build-arg HAILO_DFC_WHEEL=hailo_dataflow_compiler-3.xx.x-py3-none-linux_x86_64.whl \
    -t multideepc-compile:amd64 .

docker run --rm -v "$PWD/artifacts:/app/artifacts" multideepc-compile:amd64
# -> menghasilkan artifacts/multideepc_encode.hef
```

---

## (e) RUN di Raspberry Pi - kedua backend

Di Pi (sudah ada image; mis. `docker load -i multideepc-runtime-arm64.tar`):

### 1. Backend `onnx` (CPU) - lakukan INI dulu untuk bring-up

Tidak butuh NPU/HEF. Membuktikan container + REST contract jalan.

```bash
docker run --rm -p 8080:8080 \
    -e BACKEND=onnx \
    multideepc-runtime:arm64

# di terminal lain:
python3 test_client.py --zeros        # x_img = zeros (kasus CNN hilang)
python3 test_client.py                # sampel acak penuh
```

### 2. Backend `hailo` (NPU) - setelah `hailo-all` + HEF cocok terpasang

Butuh device NPU + HailoRT host yang di-*expose* ke container, dan `.hef` yang
cocok versinya.

```bash
docker run --rm -p 8080:8080 \
    --device /dev/hailo0 \
    -e BACKEND=hailo \
    -e HEF_PATH=artifacts/multideepc_encode.hef \
    -v "$PWD/artifacts/multideepc_encode.hef:/app/artifacts/multideepc_encode.hef" \
    multideepc-runtime:arm64

python3 test_client.py --zeros
```

Env var yang dipakai `app.py`: `BACKEND` (`onnx`|`hailo`), `ONNX_PATH`,
`HEF_PATH`. Port selalu **8080**.

---

## (f) Prasyarat di Pi + catatan VERSION-MATCH (KRITIS)

```bash
# 1. install stack Hailo (driver + HailoRT + tools)
sudo apt update
sudo apt install hailo-all
sudo reboot

# 2. aktifkan PCIe Gen3 (boost throughput NPU)
#    di /boot/firmware/config.txt tambahkan baris:
#       dtparam=pciex1_gen=3
sudo reboot

# 3. cek device + VERSI HailoRT/firmware
hailortcli fw-control identify
```

> ### ⚠️ VERSION-MATCH (penyebab kegagalan paling umum)
> Versi **HailoRT** di Pi HARUS cocok dengan versi **DFC** yang dipakai untuk
> meng-compile `.hef`. Jika tidak cocok, Pi akan menolak load HEF (error saat
> `configure`/`activate`). Catat output `hailortcli fw-control identify`
> SEBELUM mencoba load `.hef`, dan samakan versi DFC di `Dockerfile.compile`
> (`--build-arg HAILO_DFC_WHEEL=...`) dengan versi HailoRT itu.

---

## (g) Checklist - kirim balik ke kami

Setelah mencoba di Pi, kirimkan:

- [ ] **Versi HailoRT/firmware** di Pi - output lengkap `hailortcli fw-control identify`.
- [ ] Konfirmasi backend `onnx` jalan: output `python3 test_client.py --zeros`
      (`/health` + JSON `/predict`).
- [ ] Status `dtparam=pciex1_gen=3` (PCIe Gen3 aktif atau tidak).
- [ ] Semua **error load HEF** untuk backend `hailo` (log `configure`/`activate`
      dari HailoRT), jika ada.
- [ ] Versi/tag image runtime yang dijalankan (mis. `multideepc-runtime:arm64`).

Dengan versi HailoRT dari Pi, kami akan meng-compile `.hef` dengan DFC yang
cocok dan mengirimkannya kembali.
