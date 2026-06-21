# Step 2 - Scaler recovery: BLOCKED (evidence-based)

Goal: recover the per-branch StandardScaler (mean/std) used at training so
inference inputs are normalized identically. Without it, model outputs are wrong
and Hailo INT8 calibration is wrong.

## What we have
- Raw per-grid NDVI stats: `Data NDVI Pergrid/*.xlsx` (13 files, 26005 grids total)
- Raw per-inspection-point sensor: `Data Sensor/*.xlsx` (13 files, ~16 points/file)
- Trained weights + outputs: `Z_latent.npy` (26005,64), `Q_soft.npy`, predictions CSV

## Why a *correct, verified* scaler cannot be produced from this project

1. **Assembled input matrix X was never saved.** `hasil_clustering_pergrid.csv`
   contains only grid_id, coords, cluster predictions, confidence - NO feature
   values. So we cannot read back the exact [26005, 88] inputs that produced
   `Z_latent.npy`.

2. **img branch (64 feat) is unrecoverable.** Those features come from a separate
   CNN extractor (`output_cnn_nirred/`) that does not exist in the project, and no
   img input data is stored. No CNN -> no img features -> no img scaler, and no way
   to reproduce the full encode path.

3. **Feature selection/order is ambiguous.** The raw columns over-supply each branch:
   - ndvi branch needs **7**, raw NDVI has **9** stats
     (Mean, Min, Max, Std, Median, Variance, P25, P50, P75) - which 7? which order?
   - env branch needs **5**, raw "Sensor Lingkungan" has **6**
     (Suhu, Humidity, CO2, NH3, CO, NO2) - which 5?
   - soil branch needs **7**, raw "Sensor 7 in 1" has **7**
     (N, P, K, EC, Suhu, Humidity, PH) - exact count match (only confident branch).
   - spat branch needs **5**, raw has Long/Lat (+ area/perimeter?) - composition unknown.
   Different choices/orderings yield different scalers and different model outputs.

4. **No verification path.** A re-derived scaler could normally be validated by
   running `encode(X)` and comparing to `Z_latent.npy`. That check is impossible here
   because (2) the img branch is missing, so the full encode cannot be reproduced.

Conclusion: any scaler reconstructed now would be an unverifiable guess. Shipping it
would silently corrupt every inference and the Hailo calibration. Not delivered.

## The clean fix (one of these unblocks Step 2 immediately)
Request from whoever trained the model (preferred - fastest, exact):
- **A.** The saved scaler object(s): `scaler_ndvi/env/soil/spat(.pkl/.npy)` (mean_, scale_).
- **B.** The training feature-assembly script (how raw xlsx -> [N,88] X, column order,
  grid<->zone join, train-set selection) AND the CNN extractor `output_cnn_nirred/`.
- **C.** The saved training input matrix `X_train.npy` (then we re-fit StandardScaler and
  verify against `Z_latent.npy`, given the CNN).

Note: even with A/C, end-to-end on-device inference still needs the **CNN extractor**
to turn a new image patch into the 64-dim img features. That is a separate missing
artifact (see Step 3) and must be retrieved regardless.

## Best-effort scaffold (NOT a substitute for the above)
`build_features.py` can stub the grid<->zone join + per-branch assembly so that, once
the real column mapping + CNN are supplied, fitting `StandardScaler` and verifying
against `Z_latent.npy` is a few minutes' work. It is intentionally NOT auto-run as the
production scaler because its column choices are unverified guesses.
