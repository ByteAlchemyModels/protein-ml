"""
Phase 1: ESM-2 Hemolysis Predictor
==================================

This script:
  - Loads the Phase 0 hemolysis dataset
  - Trains/validates a transformer-based classifier using ESM-2 embeddings
  - Saves:
      * fine-tuned classifier weights
      * a reusable embedding extractor
      * metrics and training curves

Model:
  - Backbone: facebook/esm2_t12_35M_UR50D (Hugging Face)
  - Head: 2-layer MLP for binary classification (hemolytic vs non-hemolytic)
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("data")
MODEL_DIR = BASE_DIR / "models" / "esm2_hemolysis_predictor"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAX_LENGTH = 128  # plenty for your peptides
BATCH_SIZE = 16   # CPU-friendly
LR = 1e-4
EPOCHS = 15
WARMUP_RATIO = 0.1


# =============================================================================
# Dataset
# =============================================================================

class HemolysisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer):
        """
        df must contain columns: sequence, hemolytic
        """
        self.sequences = df["sequence"].astype(str).tolist()
        self.labels = df["hemolytic"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # ESM expects protein sequences; we treat peptides the same way.
        encoded = self.tokenizer(
            seq,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        return item


# =============================================================================
# Model
# =============================================================================

class ESMHemolysisClassifier(nn.Module):
    def __init__(self, esm_model_name: str, dropout: float = 0.2, train_backbone: bool = False):
        """
        Wrapper around ESM-2 for sequence-level classification.

        train_backbone:
            - False: freeze all ESM parameters, train only head (CPU-friendly)
            - True: fine-tune last layers (requires more compute)
        """
        super().__init__()
        self.esm = AutoModel.from_pretrained(esm_model_name)
        hidden_size = self.esm.config.hidden_size

        if not train_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Returns:
            logits: (batch, 1)
        """
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # ESM-2 uses the first token ([CLS]-like) for sequence representation
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)  # (batch, 1)
        return logits

    def extract_embeddings(self, input_ids, attention_mask=None):
        """
        Reusable sequence-level embedding (before classification head).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb


# =============================================================================
# Training / Evaluation Utilities
# =============================================================================

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.stack([b[k] for b in batch], dim=0)
    return collated


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    pos_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).unsqueeze(1)  # (batch, 1)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0)
    return all_logits, all_labels


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "pos_rate": float(labels.mean()),
    }


# =============================================================================
# Data Loading for Phase 1
# =============================================================================

def load_phase0_hemolysis(path: Path) -> pd.DataFrame:
    """
    Load the Phase 0 hemolysis CSV.

    Expected columns include:
      - sequence (str)
      - hemolytic (0/1)
      - split (train/val/test) if available (optional)
    """
    df = pd.read_csv(path)
    if "sequence" not in df.columns or "hemolytic" not in df.columns:
        raise ValueError(f"{path} must contain 'sequence' and 'hemolytic' columns")

    # If a 'split' column exists, use it; otherwise, do random split.
    if "split" in df.columns:
        print(f"[INFO] Using predefined split from column 'split' in {path.name}")
        return df
    else:
        print(f"[INFO] No 'split' column in {path.name}, will create random train/val/test splits")
        return df


def create_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create or use existing train/val/test splits.
    """
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        # If no val split exists, carve from train
        if len(val_df) == 0:
            train_df, val_df = train_test_split(
                train_df,
                test_size=0.15,
                random_state=RANDOM_SEED,
                stratify=train_df["hemolytic"],
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=RANDOM_SEED,
            stratify=df["hemolytic"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_df["hemolytic"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    print(f"[SPLITS] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


# =============================================================================
# Main Training Logic
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 1: ESM-2 HEMOLYSIS PREDICTOR")
    print("=" * 70)

    hem_path = BASE_DIR / "amp_hemolysis_sample_1104.csv"
    if not hem_path.exists():
        raise FileNotFoundError(
            f"{hem_path} not found. Make sure you ran Phase 0 and/or update the path."
        )

    df = load_phase0_hemolysis(hem_path)
    train_df, val_df, test_df = create_splits(df)

    print(f"[LABELS] Train pos rate: {train_df['hemolytic'].mean():.3f}")
    print(f"[LABELS]  Val  pos rate: {val_df['hemolytic'].mean():.3f}")
    print(f"[LABELS] Test pos rate: {test_df['hemolytic'].mean():.3f}")

    print("\n[MODEL] Loading tokenizer and model:", ESM_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = ESMHemolysisClassifier(
        esm_model_name=ESM_MODEL_NAME,
        dropout=0.2,
        train_backbone=False,  # start by freezing backbone on CPU
    ).to(DEVICE)

    train_ds = HemolysisDataset(train_df, tokenizer)
    val_ds = HemolysisDataset(val_df, tokenizer)
    test_ds = HemolysisDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Handle class imbalance: compute pos_weight for BCEWithLogitsLoss
    pos_rate = train_df["hemolytic"].mean()
    if pos_rate > 0 and pos_rate < 1:
        pos_weight = (1 - pos_rate) / pos_rate
    else:
        pos_weight = 1.0
    print(f"\n[TRAIN] Positive class rate: {pos_rate:.3f}, pos_weight: {pos_weight:.3f}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-2,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_auc = -np.inf
    history = []

    print("\n[TRAINING] Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, pos_weight=pos_weight
        )

        # Train metrics
        train_logits, train_labels = predict(model, train_loader, DEVICE)
        train_metrics = compute_metrics(train_logits, train_labels)

        # Val metrics
        val_logits, val_labels = predict(model, val_loader, DEVICE)
        val_metrics = compute_metrics(val_logits, val_labels)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train AUC: {train_metrics['roc_auc']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_path = MODEL_DIR / "model_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] New best model saved to {best_path}")

    # Final evaluation on test set using best model
    print("\n[TEST] Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    test_logits, test_labels = predict(model, test_loader, DEVICE)
    test_metrics = compute_metrics(test_logits, test_labels)

    print("\n[TEST METRICS]")
    for k, v in test_metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{np.array(v)}")
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save final model (same as best) and config/metrics
    final_model_path = MODEL_DIR / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)

    config = {
        "esm_model_name": ESM_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "device": str(DEVICE),
        "pos_weight": float(pos_weight),
        "random_seed": RANDOM_SEED,
    }

    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(MODEL_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n[SAVE] Model, config, history, and test metrics saved in:", MODEL_DIR)

    print("\nNext steps:")
    print("  - Use ESMHemolysisClassifier.extract_embeddings(...) as reusable embedding extractor")
    print("  - Or wrap this model in a scoring function for the closed-loop pipeline later")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
