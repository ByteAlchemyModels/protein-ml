"""
Phase 2: Conditional Peptide Generator (ESM-2 + Property-Conditioned Decoder)
=============================================================================
This script:
- Loads the Phase 0 hemolysis dataset (amp_hemolysis_sample_1104.csv)
- Trains a conditional sequence generator that produces peptide sequences
  conditioned on desired property labels (hemolytic, soluble_rule)
- Architecture:
    * Encoder: frozen ESM-2 backbone to extract context-aware AA embeddings
    * Decoder: lightweight Transformer decoder with property-conditioning tokens
    * Training: teacher-forced autoregressive next-token prediction
- Saves:
    * trained generator weights
    * generation config and training history
    * a batch of conditionally-generated peptide sequences for inspection

Approach:
  We prepend special "condition tokens" to each sequence before training.
  The model learns to generate peptides whose properties match the requested
  condition.  At inference time we supply the desired condition prefix and
  let the decoder sample autoregressively.

Metrics tracked during training (essentially zero extra cost):
  - Token-level train accuracy (top-1 and top-5)
  - Train perplexity (exp of mean cross-entropy; e.g. loss ~2.5 → PPL ~12,
    meaning the model effectively chooses among ~12 equally-likely tokens)
  - Per-condition accuracy and perplexity at evaluation time so you can
    spot if the underrepresented (hemo=0, sol=0) bucket is lagging

Uses ONLY packages from environment.yml (torch, transformers, pandas, numpy,
scikit-learn, matplotlib, seaborn).
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_transformers")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR       = Path("data")
MODEL_DIR      = BASE_DIR / "models" / "esm2_conditional_generator"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED    = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAX_LENGTH     = 128          # max peptide length (tokens incl. special)
BATCH_SIZE     = 8            # CPU-friendly
LR             = 5e-4
EPOCHS         = 20
WARMUP_RATIO   = 0.1
DECODER_LAYERS = 4
DECODER_HEADS  = 4
DECODER_DIM    = 480          # match ESM-2 t12 hidden size
DECODER_FF_DIM = 1024
DROPOUT        = 0.1
TEMPERATURE    = 1.0          # sampling temperature at generation
TOP_K          = 20           # top-k sampling
NUM_GENERATE   = 50           # sequences to generate in final demo

# Condition tokens  – we map (hemolytic, soluble_rule) combos to special tags
# These are prepended to the sequence before tokenisation so the model
# learns to associate the label with the sequence distribution.
CONDITION_MAP = {
    (0, 0): "<NON_HEMO|INSOL>",
    (0, 1): "<NON_HEMO|SOL>",
    (1, 0): "<HEMO|INSOL>",
    (1, 1): "<HEMO|SOL>",
}

# Reverse lookup: condition-token index → (hemolytic, soluble_rule)
COND_IDX_TO_LABEL = {}

# We will build a small custom vocabulary on top of ESM tokens for conditions.
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + list(CONDITION_MAP.values())
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
TOK2IDX = {t: i for i, t in enumerate(VOCAB)}
IDX2TOK = {i: t for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = TOK2IDX["<PAD>"]
BOS_IDX = TOK2IDX["<BOS>"]
EOS_IDX = TOK2IDX["<EOS>"]
UNK_IDX = TOK2IDX["<UNK>"]

# Populate reverse lookup
for (h, s), tag in CONDITION_MAP.items():
    COND_IDX_TO_LABEL[TOK2IDX[tag]] = (h, s)


# =============================================================================
# Tokeniser helpers (lightweight, no new packages)
# =============================================================================

def encode_sequence(seq: str, hemolytic: int, soluble: int,
                    max_len: int = MAX_LENGTH) -> List[int]:
    """
    Encode a peptide into token IDs:
        [COND] [BOS] AA1 AA2 ... AAn [EOS] [PAD...]
    """
    cond_tag = CONDITION_MAP.get((int(hemolytic), int(soluble)), "<UNK>")
    tokens = [TOK2IDX[cond_tag], BOS_IDX]
    for aa in seq.upper():
        tokens.append(TOK2IDX.get(aa, UNK_IDX))
    tokens.append(EOS_IDX)
    # Truncate if too long
    tokens = tokens[:max_len]
    # Pad
    tokens += [PAD_IDX] * (max_len - len(tokens))
    return tokens


def decode_tokens(token_ids: List[int], strip_special: bool = True) -> str:
    """Decode a list of token IDs back into a peptide string."""
    result = []
    for tid in token_ids:
        tok = IDX2TOK.get(tid, "?")
        if strip_special and tok in SPECIAL_TOKENS:
            continue
        result.append(tok)
    return "".join(result)


# =============================================================================
# Dataset
# =============================================================================

class PeptideGenerationDataset(Dataset):
    """
    Each item returns (input_ids, target_ids, cond_label) where target is input
    shifted right by one (standard autoregressive LM objective).
    cond_label is (hemolytic, soluble_rule) for per-condition metric tracking.
    """

    def __init__(self, df: pd.DataFrame, max_len: int = MAX_LENGTH):
        self.data: List[List[int]] = []
        self.cond_labels: List[Tuple[int, int]] = []
        for _, row in df.iterrows():
            ids = encode_sequence(
                row["sequence"],
                row["hemolytic"],
                row.get("soluble_rule", 0),
                max_len=max_len,
            )
            self.data.append(ids)
            self.cond_labels.append((int(row["hemolytic"]), int(row.get("soluble_rule", 0))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        input_ids  = torch.tensor(ids[:-1], dtype=torch.long)   # everything except last
        target_ids = torch.tensor(ids[1:],  dtype=torch.long)   # shifted by 1
        cond_label = torch.tensor(self.cond_labels[idx], dtype=torch.long)  # (2,)
        return input_ids, target_ids, cond_label


def collate_gen(batch):
    inputs  = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    conds   = torch.stack([b[2] for b in batch])
    return inputs, targets, conds


# =============================================================================
# ESM-2 Embedding Extractor (reused from Phase 1 concept)
# =============================================================================

class ESMEmbeddingExtractor(nn.Module):
    """
    Frozen ESM-2 backbone that provides per-residue embeddings.
    These are projected into the decoder dimension and used as *memory*
    for the cross-attention in the Transformer decoder.
    """

    def __init__(self, esm_model_name: str, decoder_dim: int):
        super().__init__()
        self.esm = AutoModel.from_pretrained(esm_model_name)
        esm_hidden = self.esm.config.hidden_size
        for param in self.esm.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(esm_hidden, decoder_dim)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        """Return projected per-residue embeddings (batch, seq_len, decoder_dim)."""
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state           # (B, L, esm_hidden)
        return self.proj(hidden)                     # (B, L, decoder_dim)


# =============================================================================
# Conditional Decoder (Transformer-based autoregressive LM)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConditionalPeptideDecoder(nn.Module):
    """
    Lightweight Transformer decoder language model.

    Inputs:
        token_ids – (batch, seq_len) integer token IDs from our custom vocab
        esm_memory – (batch, mem_len, decoder_dim) optional ESM-2 embeddings
                     for cross-attention (provides protein-LM context)

    The condition tag is simply the first token in the sequence, so the model
    implicitly conditions on the desired property via causal attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = DECODER_DIM,
        nhead: int = DECODER_HEADS,
        num_layers: int = DECODER_LAYERS,
        dim_feedforward: int = DECODER_FF_DIM,
        dropout: float = DROPOUT,
        max_len: int = MAX_LENGTH,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc    = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, token_ids: torch.Tensor,
                esm_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids:  (batch, seq_len)
            esm_memory: (batch, mem_len, d_model) or None

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Embeddings
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(S, device)

        # Padding mask
        tgt_key_padding_mask = (token_ids == self.pad_idx)

        if esm_memory is not None:
            out = self.transformer_decoder(
                tgt=x,
                memory=esm_memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        else:
            # Self-attention only (no cross-attention memory)
            # Use a dummy memory of zeros so the standard decoder API works
            dummy_mem = torch.zeros(B, 1, self.d_model, device=device)
            out = self.transformer_decoder(
                tgt=x,
                memory=dummy_mem,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        logits = self.output_proj(out)  # (B, S, vocab_size)
        return logits


# =============================================================================
# Full Generator Model (wraps ESM encoder + decoder)
# =============================================================================

class ConditionalPeptideGenerator(nn.Module):
    """
    End-to-end conditional peptide generator.

    Training mode:
        Given a labelled peptide, encode the raw AA sequence through frozen ESM-2
        to get rich per-residue features, then decode autoregressively with the
        condition prefix using our lightweight Transformer decoder.

    Generation mode:
        Supply a condition prefix, optionally provide an ESM-2 encoded "seed"
        peptide for cross-attention context, then sample autoregressively.
    """

    def __init__(self, esm_model_name: str, vocab_size: int, **decoder_kwargs):
        super().__init__()
        self.esm_encoder = ESMEmbeddingExtractor(esm_model_name, decoder_kwargs.get("d_model", DECODER_DIM))
        self.decoder = ConditionalPeptideDecoder(vocab_size=vocab_size, **decoder_kwargs)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

    def _esm_encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Tokenise raw AA strings and run through frozen ESM-2."""
        encoded = self.esm_tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(next(self.esm_encoder.parameters()).device)
        attention_mask = encoded["attention_mask"].to(input_ids.device)
        return self.esm_encoder(input_ids, attention_mask)

    def forward(self, token_ids: torch.Tensor,
                sequences: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            token_ids:  (batch, seq_len) – custom-vocab encoded tokens
            sequences:  list[str] – raw AA strings for ESM encoding (optional)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        esm_memory = None
        if sequences is not None:
            esm_memory = self._esm_encode_sequences(sequences)
        return self.decoder(token_ids, esm_memory=esm_memory)

    @torch.no_grad()
    def generate(
        self,
        hemolytic: int,
        soluble: int,
        max_new_tokens: int = 60,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
        seed_sequence: Optional[str] = None,
    ) -> str:
        """
        Autoregressively generate a peptide conditioned on properties.

        Args:
            hemolytic:     desired hemolytic label (0 or 1)
            soluble:       desired soluble_rule label (0 or 1)
            max_new_tokens: max amino acids to generate
            temperature:   sampling temperature
            top_k:         top-k sampling
            seed_sequence: optional AA string for ESM cross-attention context

        Returns:
            generated peptide string (amino acids only)
        """
        self.eval()
        device = next(self.parameters()).device

        cond_tag = CONDITION_MAP.get((int(hemolytic), int(soluble)), "<UNK>")
        cond_idx = TOK2IDX[cond_tag]

        # Start with [COND] [BOS]
        generated = [cond_idx, BOS_IDX]

        # Optionally get ESM memory from seed
        esm_memory = None
        if seed_sequence is not None:
            esm_memory = self._esm_encode_sequences([seed_sequence])

        for _ in range(max_new_tokens):
            inp = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.decoder(inp, esm_memory=esm_memory)  # (1, cur_len, V)
            next_logits = logits[0, -1, :] / temperature       # (V,)

            # Top-k filtering
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, top_k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(0, topk_idx, topk_vals)
                next_logits = mask

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == EOS_IDX:
                break
            if next_token == PAD_IDX:
                break

            generated.append(next_token)

        return decode_tokens(generated, strip_special=True)


# =============================================================================
# Training utilities
# =============================================================================

def train_one_epoch(
    model: ConditionalPeptideGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    sequences_list: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """
    Standard teacher-forced training step.

    Accumulates token-level top-1 accuracy, top-5 accuracy, and cross-entropy
    inside the forward pass at essentially zero extra cost.

    Returns dict with: loss, accuracy_top1, accuracy_top5, perplexity
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="sum")

    total_loss     = 0.0
    total_correct1 = 0
    total_correct5 = 0
    total_tokens   = 0

    for batch_idx, (input_ids, target_ids, cond_labels) in enumerate(dataloader):
        input_ids  = input_ids.to(device)
        target_ids = target_ids.to(device)

        raw_seqs = None
        if sequences_list is not None:
            raw_seqs = sequences_list[batch_idx]

        logits = model(input_ids, sequences=raw_seqs)  # (B, S, V)

        B, S, V = logits.shape
        flat_logits = logits.reshape(B * S, V)
        flat_targets = target_ids.reshape(B * S)

        loss = criterion(flat_logits, flat_targets)

        # --- Accumulate accuracy inside forward pass (zero extra cost) ---
        non_pad = flat_targets != PAD_IDX
        n_tokens = non_pad.sum().item()

        # Top-1
        preds_top1 = flat_logits.argmax(dim=-1)
        correct1 = ((preds_top1 == flat_targets) & non_pad).sum().item()

        # Top-5 (clamp k to vocab size in case V < 5)
        k5 = min(5, V)
        _, preds_top5 = flat_logits.topk(k5, dim=-1)  # (B*S, k5)
        target_expanded = flat_targets.unsqueeze(-1).expand_as(preds_top5)
        hit_top5 = (preds_top5 == target_expanded).any(dim=-1)
        correct5 = (hit_top5 & non_pad).sum().item()

        total_loss     += loss.item()
        total_correct1 += correct1
        total_correct5 += correct5
        total_tokens   += n_tokens

        # Normalise loss for backward (sum → mean)
        loss_normed = loss / max(n_tokens, 1)
        optimizer.zero_grad()
        loss_normed.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / max(total_tokens, 1)
    acc_top1 = total_correct1 / max(total_tokens, 1)
    acc_top5 = total_correct5 / max(total_tokens, 1)
    ppl      = math.exp(min(avg_loss, 100))  # clamp to avoid overflow

    return {
        "loss": avg_loss,
        "accuracy_top1": acc_top1,
        "accuracy_top5": acc_top5,
        "perplexity": ppl,
    }


@torch.no_grad()
def evaluate(
    model: ConditionalPeptideGenerator,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute validation/test loss, top-1 accuracy, top-5 accuracy, and
    perplexity.  Also reports **per-condition** metrics so you can spot
    whether underrepresented conditions (e.g. hemo=0, sol=0 with ~84
    training examples) are lagging.

    Returns:
        overall_metrics: dict with loss, accuracy_top1, accuracy_top5, perplexity
        per_condition:   dict keyed by condition tag string, each containing
                         the same four metric keys
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="none")

    # Global accumulators
    total_loss     = 0.0
    total_correct1 = 0
    total_correct5 = 0
    total_tokens   = 0

    # Per-condition accumulators: keyed by (hemolytic, soluble_rule) tuple
    cond_loss     = {}
    cond_correct1 = {}
    cond_correct5 = {}
    cond_tokens   = {}

    for input_ids, target_ids, cond_labels in dataloader:
        input_ids  = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # (B, S, V)
        B, S, V = logits.shape

        flat_logits  = logits.reshape(B * S, V)
        flat_targets = target_ids.reshape(B * S)

        # Per-token loss (unreduced)
        per_token_loss = criterion(flat_logits, flat_targets).reshape(B, S)

        # Top-1 preds
        preds_top1 = flat_logits.argmax(dim=-1).reshape(B, S)

        # Top-5 preds
        k5 = min(5, V)
        _, preds_top5 = flat_logits.topk(k5, dim=-1)          # (B*S, k5)
        preds_top5 = preds_top5.reshape(B, S, k5)
        target_exp = target_ids.unsqueeze(-1).expand_as(preds_top5)
        hit_top5   = (preds_top5 == target_exp).any(dim=-1)    # (B, S)

        # Non-pad mask per sample
        non_pad = target_ids != PAD_IDX  # (B, S)

        for i in range(B):
            mask_i    = non_pad[i]
            n_tok     = mask_i.sum().item()
            loss_i    = per_token_loss[i][mask_i].sum().item()
            correct1_i = ((preds_top1[i] == target_ids[i]) & mask_i).sum().item()
            correct5_i = (hit_top5[i] & mask_i).sum().item()

            total_loss     += loss_i
            total_correct1 += correct1_i
            total_correct5 += correct5_i
            total_tokens   += n_tok

            # Per-condition
            key = (cond_labels[i, 0].item(), cond_labels[i, 1].item())
            cond_loss.setdefault(key, 0.0)
            cond_correct1.setdefault(key, 0)
            cond_correct5.setdefault(key, 0)
            cond_tokens.setdefault(key, 0)
            cond_loss[key]     += loss_i
            cond_correct1[key] += correct1_i
            cond_correct5[key] += correct5_i
            cond_tokens[key]   += n_tok

    # --- Aggregate overall ---
    avg_loss = total_loss / max(total_tokens, 1)
    overall = {
        "loss":          avg_loss,
        "accuracy_top1": total_correct1 / max(total_tokens, 1),
        "accuracy_top5": total_correct5 / max(total_tokens, 1),
        "perplexity":    math.exp(min(avg_loss, 100)),
    }

    # --- Aggregate per condition ---
    per_condition = {}
    for key in sorted(cond_loss.keys()):
        n = cond_tokens[key]
        c_loss = cond_loss[key] / max(n, 1)
        tag = CONDITION_MAP.get(key, str(key))
        per_condition[tag] = {
            "n_samples":      n,
            "loss":           c_loss,
            "accuracy_top1":  cond_correct1[key] / max(n, 1),
            "accuracy_top5":  cond_correct5[key] / max(n, 1),
            "perplexity":     math.exp(min(c_loss, 100)),
        }

    return overall, per_condition


# =============================================================================
# Sequence quality analysis
# =============================================================================

def analyse_generated_sequences(
    generated: List[Dict],
    real_df: pd.DataFrame,
) -> Dict:
    """
    Compute basic quality metrics for generated peptides:
        - length distribution vs real
        - amino acid composition similarity
        - fraction of valid AA-only sequences
        - novelty (fraction not in training set)
    """
    gen_seqs = [g["sequence"] for g in generated]
    real_seqs = set(real_df["sequence"].tolist())

    lengths   = [len(s) for s in gen_seqs]
    valid     = [all(aa in AMINO_ACIDS for aa in s) and len(s) > 0 for s in gen_seqs]
    novel     = [s not in real_seqs for s in gen_seqs]

    # AA frequency
    gen_aa_counts  = {}
    real_aa_counts = {}
    for s in gen_seqs:
        for aa in s:
            gen_aa_counts[aa] = gen_aa_counts.get(aa, 0) + 1
    for s in real_df["sequence"]:
        for aa in s:
            real_aa_counts[aa] = real_aa_counts.get(aa, 0) + 1

    # Normalise
    gen_total  = max(sum(gen_aa_counts.values()), 1)
    real_total = max(sum(real_aa_counts.values()), 1)
    gen_freq   = {aa: gen_aa_counts.get(aa, 0) / gen_total for aa in AMINO_ACIDS}
    real_freq  = {aa: real_aa_counts.get(aa, 0) / real_total for aa in AMINO_ACIDS}

    return {
        "n_generated": len(gen_seqs),
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
        "valid_fraction": float(np.mean(valid)),
        "novel_fraction": float(np.mean(novel)),
        "aa_freq_generated": gen_freq,
        "aa_freq_real": real_freq,
    }


# =============================================================================
# Plotting
# =============================================================================

def create_training_plots(history: List[Dict], save_dir: Path) -> None:
    """Save training curves including loss, accuracy (top-1/top-5), and perplexity."""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    train_acc1 = [h["train_accuracy_top1"] for h in history]
    val_acc1   = [h["val_accuracy_top1"] for h in history]
    train_acc5 = [h["train_accuracy_top5"] for h in history]
    val_acc5   = [h["val_accuracy_top5"] for h in history]
    train_ppl  = [h["train_perplexity"] for h in history]
    val_ppl    = [h["val_perplexity"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 2 – Conditional Peptide Generator Training", fontsize=14, fontweight="bold")

    # Loss
    axes[0, 0].plot(epochs, train_loss, label="Train", marker="o", markersize=4)
    axes[0, 0].plot(epochs, val_loss,   label="Val",   marker="s", markersize=4)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Top-1 Accuracy
    axes[0, 1].plot(epochs, train_acc1, label="Train Top-1", marker="o", markersize=4)
    axes[0, 1].plot(epochs, val_acc1,   label="Val Top-1",   marker="s", markersize=4)
    axes[0, 1].plot(epochs, train_acc5, label="Train Top-5", marker="^", markersize=4, linestyle="--")
    axes[0, 1].plot(epochs, val_acc5,   label="Val Top-5",   marker="d", markersize=4, linestyle="--")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Token Accuracy")
    axes[0, 1].set_title("Top-1 & Top-5 Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Perplexity
    axes[1, 0].plot(epochs, train_ppl, label="Train PPL", marker="o", markersize=4, color="purple")
    axes[1, 0].plot(epochs, val_ppl,   label="Val PPL",   marker="s", markersize=4, color="orange")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Perplexity")
    axes[1, 0].set_title("Perplexity (exp of CE loss)"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Per-condition val accuracy (final epoch)
    if "val_per_condition" in history[-1]:
        cond_data = history[-1]["val_per_condition"]
        tags = list(cond_data.keys())
        acc1_vals = [cond_data[t]["accuracy_top1"] for t in tags]
        acc5_vals = [cond_data[t]["accuracy_top5"] for t in tags]
        x = np.arange(len(tags))
        width = 0.35
        axes[1, 1].bar(x - width/2, acc1_vals, width, label="Top-1", color="steelblue")
        axes[1, 1].bar(x + width/2, acc5_vals, width, label="Top-5", color="salmon")
        axes[1, 1].set_xlabel("Condition"); axes[1, 1].set_ylabel("Token Accuracy")
        axes[1, 1].set_title("Per-Condition Val Accuracy (final epoch)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([t.replace("<","").replace(">","") for t in tags],
                                    rotation=20, ha="right", fontsize=8)
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3, axis="y")
    else:
        axes[1, 1].text(0.5, 0.5, "No per-condition data", ha="center", va="center")
        axes[1, 1].set_title("Per-Condition Val Accuracy")

    plt.tight_layout()
    plt.savefig(save_dir / "phase2_training_curves.png", dpi=100, bbox_inches="tight")
    print(f"  Saved: {save_dir / 'phase2_training_curves.png'}")
    plt.close()


def create_generation_plots(analysis: Dict, save_dir: Path) -> None:
    """Plot AA frequency comparison between generated and real peptides."""
    save_dir.mkdir(parents=True, exist_ok=True)

    gen_freq  = analysis["aa_freq_generated"]
    real_freq = analysis["aa_freq_real"]

    aas = sorted(AMINO_ACIDS)
    x = np.arange(len(aas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [real_freq.get(aa, 0) for aa in aas], width, label="Real", color="steelblue")
    ax.bar(x + width/2, [gen_freq.get(aa, 0)  for aa in aas], width, label="Generated", color="salmon")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Frequency")
    ax.set_title("AA Frequency: Real vs Generated Peptides")
    ax.set_xticks(x)
    ax.set_xticklabels(aas)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "phase2_aa_frequency.png", dpi=100, bbox_inches="tight")
    print(f"  Saved: {save_dir / 'phase2_aa_frequency.png'}")
    plt.close()


# =============================================================================
# Data Loading (mirrors Phase 1 conventions)
# =============================================================================

def load_phase0_data(path: Path) -> pd.DataFrame:
    """
    Load Phase 0 CSV.  Expected columns: sequence, hemolytic, soluble_rule.
    If soluble_rule is missing we create it with a default of 0.
    """
    df = pd.read_csv(path)
    required = {"sequence", "hemolytic"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{path} must contain columns: {required}")

    if "soluble_rule" not in df.columns:
        print("[WARN] 'soluble_rule' column not found – defaulting to 0")
        df["soluble_rule"] = 0

    # Filter sequences with non-standard AAs
    valid_mask = df["sequence"].apply(
        lambda s: all(aa in AMINO_ACIDS for aa in str(s).upper()) and len(str(s)) > 0
    )
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        print(f"[DATA] Removed {n_removed} sequences with non-standard amino acids")
    df = df[valid_mask].reset_index(drop=True)
    df["sequence"] = df["sequence"].str.upper()
    return df


def create_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train / val / test splits."""
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)
        # If no val, carve from train
        if len(val_df) == 0:
            train_df, val_df = train_test_split(
                train_df, test_size=0.15, random_state=RANDOM_SEED,
                stratify=train_df["hemolytic"],
            )
            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_SEED, stratify=df["hemolytic"],
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["hemolytic"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

    print(f"[SPLITS] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PHASE 2: CONDITIONAL PEPTIDE GENERATOR")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    hem_path = BASE_DIR / "amp_hemolysis_sample_1104.csv"
    if not hem_path.exists():
        raise FileNotFoundError(
            f"{hem_path} not found.  Run Phase 0 first or place the CSV under {BASE_DIR}/"
        )

    df = load_phase0_data(hem_path)
    print(f"[DATA] Loaded {len(df)} valid peptide sequences from {hem_path.name}")

    train_df, val_df, test_df = create_splits(df)

    # Condition distribution
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        cond_counts = split.groupby(["hemolytic", "soluble_rule"]).size()
        print(f"  {name} condition distribution:\n{cond_counts.to_string()}")

    # ------------------------------------------------------------------
    # 2. Build datasets & loaders
    # ------------------------------------------------------------------
    train_ds = PeptideGenerationDataset(train_df, max_len=MAX_LENGTH)
    val_ds   = PeptideGenerationDataset(val_df,   max_len=MAX_LENGTH)
    test_ds  = PeptideGenerationDataset(test_df,  max_len=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_gen)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_gen)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_gen)

    print(f"\n[MODEL] Building Conditional Peptide Generator...")
    print(f"  Decoder: {DECODER_LAYERS} layers, {DECODER_HEADS} heads, dim={DECODER_DIM}")
    print(f"  Vocab size: {VOCAB_SIZE}  (AA + special + condition tokens)")
    print(f"  ESM-2 backbone: {ESM_MODEL_NAME} (frozen, used at generation time)")
    print(f"  Device: {DEVICE}")

    model = ConditionalPeptideGenerator(
        esm_model_name=ESM_MODEL_NAME,
        vocab_size=VOCAB_SIZE,
        d_model=DECODER_DIM,
        nhead=DECODER_HEADS,
        num_layers=DECODER_LAYERS,
        dim_feedforward=DECODER_FF_DIM,
        dropout=DROPOUT,
        max_len=MAX_LENGTH,
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} total")

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-2,
    )
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    history = []
    best_val_loss = float("inf")

    print(f"\n[TRAINING] Starting training for {EPOCHS} epochs...")
    print(f"  {'Epoch':>5} | {'TrLoss':>7} {'TrAcc1':>7} {'TrAcc5':>7} {'TrPPL':>7} | "
          f"{'VaLoss':>7} {'VaAcc1':>7} {'VaAcc5':>7} {'VaPPL':>7} | {'Time':>5}")
    print(f"  {'─'*5:>5} | {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} | "
          f"{'─'*7:>7} {'─'*7:>7} {'─'*7:>7} {'─'*7:>7} | {'─'*5:>5}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE,
            sequences_list=None,  # decoder-only for CPU efficiency
        )

        val_overall, val_per_cond = evaluate(model, val_loader, DEVICE)
        elapsed = time.time() - t0

        print(
            f"  {epoch:5d} | "
            f"{train_metrics['loss']:7.4f} {train_metrics['accuracy_top1']:7.4f} "
            f"{train_metrics['accuracy_top5']:7.4f} {train_metrics['perplexity']:7.1f} | "
            f"{val_overall['loss']:7.4f} {val_overall['accuracy_top1']:7.4f} "
            f"{val_overall['accuracy_top5']:7.4f} {val_overall['perplexity']:7.1f} | "
            f"{elapsed:5.1f}s"
        )

        # Log per-condition metrics every 5 epochs + final epoch
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"         Per-condition val metrics (epoch {epoch}):")
            for tag, cmetrics in val_per_cond.items():
                tag_short = tag.replace("<","").replace(">","")
                print(
                    f"           {tag_short:20s}  "
                    f"Acc1={cmetrics['accuracy_top1']:.4f}  "
                    f"Acc5={cmetrics['accuracy_top5']:.4f}  "
                    f"PPL={cmetrics['perplexity']:.1f}  "
                    f"(n_tokens={cmetrics['n_samples']})"
                )

        history.append({
            "epoch":               epoch,
            "train_loss":          train_metrics["loss"],
            "train_accuracy_top1": train_metrics["accuracy_top1"],
            "train_accuracy_top5": train_metrics["accuracy_top5"],
            "train_perplexity":    train_metrics["perplexity"],
            "val_loss":            val_overall["loss"],
            "val_accuracy_top1":   val_overall["accuracy_top1"],
            "val_accuracy_top5":   val_overall["accuracy_top5"],
            "val_perplexity":      val_overall["perplexity"],
            "val_per_condition":   val_per_cond,
        })

        if val_overall["loss"] < best_val_loss:
            best_val_loss = val_overall["loss"]
            best_path = MODEL_DIR / "generator_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"    [SAVE] New best model → {best_path}")

    # ------------------------------------------------------------------
    # 4. Final test evaluation
    # ------------------------------------------------------------------
    print("\n[TEST] Evaluating best model on test set...")
    model.load_state_dict(torch.load(MODEL_DIR / "generator_best.pt", map_location=DEVICE))
    test_overall, test_per_cond = evaluate(model, test_loader, DEVICE)

    print(f"\n  {'Metric':<20s} {'Value':>10s}")
    print(f"  {'─'*20} {'─'*10}")
    print(f"  {'Loss':<20s} {test_overall['loss']:10.4f}")
    print(f"  {'Top-1 Accuracy':<20s} {test_overall['accuracy_top1']:10.4f}")
    print(f"  {'Top-5 Accuracy':<20s} {test_overall['accuracy_top5']:10.4f}")
    print(f"  {'Perplexity':<20s} {test_overall['perplexity']:10.1f}")

    print(f"\n  Per-condition test metrics:")
    for tag, cmetrics in test_per_cond.items():
        tag_short = tag.replace("<","").replace(">","")
        print(
            f"    {tag_short:20s}  "
            f"Acc1={cmetrics['accuracy_top1']:.4f}  "
            f"Acc5={cmetrics['accuracy_top5']:.4f}  "
            f"PPL={cmetrics['perplexity']:.1f}  "
            f"(n_tokens={cmetrics['n_samples']})"
        )

    # ------------------------------------------------------------------
    # 5. Generate peptides for each condition
    # ------------------------------------------------------------------
    print(f"\n[GENERATE] Sampling {NUM_GENERATE} peptides per condition...")
    generated_all = []
    for (hemo, sol), tag in CONDITION_MAP.items():
        for _ in range(NUM_GENERATE):
            seq = model.generate(
                hemolytic=hemo, soluble=sol,
                max_new_tokens=60,
                temperature=TEMPERATURE, top_k=TOP_K,
            )
            generated_all.append({
                "condition": tag,
                "hemolytic": hemo,
                "soluble_rule": sol,
                "sequence": seq,
                "length": len(seq),
            })

    gen_df = pd.DataFrame(generated_all)
    gen_path = MODEL_DIR / "generated_peptides.csv"
    gen_df.to_csv(gen_path, index=False)
    print(f"  Saved {len(gen_df)} generated sequences → {gen_path}")

    # Show samples
    print("\n[SAMPLES] Example generated peptides:")
    for tag in CONDITION_MAP.values():
        subset = gen_df[gen_df["condition"] == tag].head(3)
        print(f"\n  {tag}:")
        for _, row in subset.iterrows():
            print(f"    {row['sequence']:<50s}  (len={row['length']})")

    # ------------------------------------------------------------------
    # 6. Quality analysis
    # ------------------------------------------------------------------
    analysis = analyse_generated_sequences(generated_all, df)
    print(f"\n[QUALITY] Generation analysis:")
    print(f"  Valid AA fraction:  {analysis['valid_fraction']:.3f}")
    print(f"  Novel fraction:     {analysis['novel_fraction']:.3f}")
    print(f"  Avg length:         {analysis['avg_length']:.1f} ± {analysis['std_length']:.1f}")

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    config = {
        "esm_model_name": ESM_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "decoder_layers": DECODER_LAYERS,
        "decoder_heads": DECODER_HEADS,
        "decoder_dim": DECODER_DIM,
        "decoder_ff_dim": DECODER_FF_DIM,
        "dropout": DROPOUT,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "vocab_size": VOCAB_SIZE,
        "vocab": VOCAB,
        "condition_map": {str(k): v for k, v in CONDITION_MAP.items()},
        "device": str(DEVICE),
        "random_seed": RANDOM_SEED,
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    test_results = {
        "test_loss":            float(test_overall["loss"]),
        "test_accuracy_top1":   float(test_overall["accuracy_top1"]),
        "test_accuracy_top5":   float(test_overall["accuracy_top5"]),
        "test_perplexity":      float(test_overall["perplexity"]),
        "test_per_condition":   test_per_cond,
        "generation_analysis": {
            k: v for k, v in analysis.items()
            if k not in ("aa_freq_generated", "aa_freq_real")
        },
    }
    with open(MODEL_DIR / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    with open(MODEL_DIR / "generation_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Plots
    plots_dir = BASE_DIR / "plots"
    create_training_plots(history, plots_dir)
    create_generation_plots(analysis, plots_dir)

    # Final model save
    final_path = MODEL_DIR / "generator_final.pt"
    torch.save(model.state_dict(), final_path)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"\n  Artifacts saved in: {MODEL_DIR}/")
    print(f"    generator_best.pt          – best checkpoint (lowest val loss)")
    print(f"    generator_final.pt         – final epoch checkpoint")
    print(f"    config.json                – model & training configuration")
    print(f"    training_history.json      – per-epoch metrics (incl. per-condition)")
    print(f"    test_metrics.json          – test set evaluation (incl. per-condition)")
    print(f"    generated_peptides.csv     – conditionally generated sequences")
    print(f"    generation_analysis.json   – quality analysis of generations")
    print(f"  Plots saved in: {plots_dir}/")
    print(f"    phase2_training_curves.png – loss, accuracy (top-1/5), PPL, per-cond")
    print(f"    phase2_aa_frequency.png    – AA composition comparison")
    print("\nNext steps:")
    print("  - Use ConditionalPeptideGenerator.generate() to sample new peptides")
    print("  - Score generated sequences with the Phase 1 hemolysis classifier")
    print("  - Proceed to Phase 3 (Diffusion Model) or Phase 4 (Property Scorers)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
