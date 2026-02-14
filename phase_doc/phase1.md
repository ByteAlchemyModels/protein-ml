# Phase 1 Script – `phase1_esm2_hemolysis_classifier.py`

Short overview of what the Phase 1 classifier does, how it uses the Phase 0 data, how it is trained, what it saves, and how to run it.

---

## 1. Purpose

Phase 1 trains an ESM‑2–based **binary hemolysis classifier** that plugs into your peptide design pipeline.

- Loads the Phase 0 hemolysis dataset from `data/amp_hemolysis_sample_1104.csv`.
- Predicts `hemolytic` vs `non‑hemolytic` for each peptide.
- Produces a reusable model and embedding extractor for later phases.

---

## 2. Model architecture

- **Backbone**: `facebook/esm2_t12_35M_UR50D` (Hugging Face ESM‑2).
- **Sequence embedding**: uses the first token (CLS‑like) from `last_hidden_state`.
- **Head**: small MLP  
  `ESM embedding → Linear → ReLU → Dropout → Linear → 1 logit`.
- **Backbone training**: by default, ESM‑2 is **frozen** (CPU‑friendly); only the head is trained.

---

## 3. Data and splits

### Expected input

- CSV file: `data/amp_hemolysis_sample_1104.csv`.
- Required columns:
  - `sequence` (string peptide sequence)
  - `hemolytic` (`0` or `1`)
- Optional column:
  - `split` (`train` / `val` / `test`)

If the file is missing, the script raises an error asking you to run Phase 0 or update the path.

### Split logic

- If `split` exists:
  - Use existing `train` / `val` / `test` rows.
- If `split` is missing:
  - Stratified on `hemolytic`:
    - 70% → train
    - 15% → val
    - 15% → test
- Prints:
  - Exact counts per split.
  - Positive class rate (`hemolytic` fraction) per split.

---

## 4. Dataset and training

### Tokenization / dataset

- Uses `AutoTokenizer.from_pretrained(ESM_MODEL_NAME)`.
- Settings:
  - `max_length = 128`, `truncation=True`, `padding='max_length'`.
- `HemolysisDataset`:
  - Returns `input_ids`, `attention_mask`, `labels` (float).

### Training setup

- Hyperparameters:
  - `BATCH_SIZE = 16`
  - `LR = 1e-4`
  - `EPOCHS = 15`
  - `WARMUP_RATIO = 0.1`
  - `RANDOM_SEED = 42`
- Loss:
  - `BCEWithLogitsLoss` with `pos_weight` from train positive rate:
    - `pos_weight = (1 - pos_rate) / pos_rate` if `0 < pos_rate < 1`, else `1.0`.
- Optimizer / scheduler:
  - `AdamW` with weight decay `1e-2`.
  - Linear warmup + decay over full training steps.

Each epoch:

- Train loop:
  - Computes average training loss.
- Metrics on train and val:
  - Accuracy
  - ROC AUC
  - Precision, recall, F1
  - Confusion matrix
- Logs a one‑line summary:
  - `Epoch XX | Train Loss | Train AUC/F1 | Val AUC/F1`.
- Tracks best **validation AUC** and saves `model_best.pt`.

---

## 5. Evaluation and saved artifacts

### Test evaluation

After training:

- Reloads `model_best.pt`.
- Evaluates on the **test** set.
- Prints:
  - Accuracy, ROC AUC, precision, recall, F1
  - Confusion matrix
  - Positive rate

### Artifacts

Saved under:

```text
data/models/esm2_hemolysis_predictor/
