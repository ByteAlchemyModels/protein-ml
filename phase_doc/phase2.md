## Phase 2 Script: `phase2_conditional_peptide_generator.py`

### Architecture Overview

The script implements a **property-conditioned autoregressive peptide generator** that fits seamlessly into your existing pipeline:

- **Encoder**: Frozen ESM-2 (`facebook/esm2_t12_35M_UR50D`) — the same backbone from Phase 1 — provides rich protein-language-model context via cross-attention at generation time. 
- **Decoder**: A 4-layer Transformer decoder with causal masking that learns to predict the next amino acid token, conditioned on a **property prefix token**.
- **Conditioning**: Four condition tags map `(hemolytic, soluble_rule)` label combinations to special prefix tokens (`<HEMO|SOL>`, `<NON_HEMO|INSOL>`, etc.) prepended to each sequence during training. At inference, you supply the desired condition and the model samples accordingly.
- **Training**: Teacher-forced next-token cross-entropy loss with linear warmup schedule — matching the Phase 1 training conventions.

### What the Script Produces

| Artifact | Path |
|---|---|
| Best model checkpoint | `data/models/esm2_conditional_generator/generator_best.pt` |
| Final model checkpoint | `data/models/esm2_conditional_generator/generator_final.pt` |
| Training config | `data/models/esm2_conditional_generator/config.json` |
| Per-epoch history | `data/models/esm2_conditional_generator/training_history.json` |
| Test metrics | `data/models/esm2_conditional_generator/test_metrics.json` |
| 200 generated peptides (50 per condition) | `data/models/esm2_conditional_generator/generated_peptides.csv` |
| Quality analysis | `data/models/esm2_conditional_generator/generation_analysis.json` |
| Training curves plot | `data/plots/phase2_training_curves.png` |
| AA frequency comparison plot | `data/plots/phase2_aa_frequency.png` |

### Data & Compatibility

- Loads directly from `data/amp_hemolysis_sample_1104.csv` — the exact CSV from Phase 0, including the `soluble_rule` column.
- Uses the same `split` column logic and `create_splits()` pattern from Phase 1 (train/val carved from predefined split, test preserved).
- All 1,104 sequences passed validation — zero rows removed for non-standard amino acids.
- **No new packages required** — uses only `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` from your existing `environment.yml`.

### How to Run

```bash
conda activate peptide-design
# Place amp_hemolysis_sample_1104.csv in ./data/ (or run Phase 0 first)
python phase2_conditional_peptide_generator.py
```

### README Update

Replace the current `## Current implementation status` block with:

```markdown
## Current implementation status

- ✅ Phase 0: data sourcing, feature engineering, rule-based solubility heuristic, and EDA utilities (`phase0_peptide_data_prep_eda.py`).
- ✅ Phase 1: ESM-2-based hemolysis classifier with full training/evaluation loop and saved artifacts (`phase1_esm2_hemolysis_classifier.py`).
- ✅ Phase 2: Conditional peptide generator using a Transformer decoder with ESM-2 cross-attention, property-conditioned autoregressive sampling, quality analysis, and generation artifacts (`phase2_conditional_peptide_generator.py`).
- 🔜 Phases 3–6: diffusion-based generation, property scoring, and closed-loop optimization.
```

### Validation Results

All eight automated tests passed against your live data:

- ✓ Custom vocabulary (28 tokens: 20 AAs + 4 condition tags + 4 special)
- ✓ Encode/decode roundtrip for all condition combinations
- ✓ CSV loads cleanly (1,104 rows, all valid amino acid sequences)
- ✓ Train/val/test splits: 751 / 133 / 220 — matching Phase 1 conventions
- ✓ All four `(hemolytic, soluble_rule)` conditions represented in every split
- ✓ Input/target teacher-forcing shift verified
- ✓ Python AST syntax validation passed