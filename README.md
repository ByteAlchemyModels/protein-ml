# Protein-ML: Peptide Modeling Pipeline

This repository contains a multi-phase pipeline for peptide and protein modeling, starting from hemolysis prediction and extending toward generative design and closed-loop optimization. It is designed as a realistic end-to-end ML project rather than a single benchmark model.

## Phase roadmap

- **Phase 0 – Setup & Data**
  - Source peptide datasets (AMPDeep-style hemolysis), engineer basic physicochemical features, and run exploratory data analysis.
  - Export CPU-friendly sampled datasets for rapid experimentation.

- **Phase 1 – Transformer Predictor**
  - Fine-tune a transformer model (ESM-2) to classify hemolytic vs non-hemolytic peptides.
  - Save model weights, metrics, and reusable sequence embeddings for downstream tasks.

- **Phase 2 – Transformer LM (planned)**
  - Conditional sequence generation conditioned on properties (e.g., hemolysis/solubility labels).
  - Use pretrained protein language models as a base for guided peptide generation.

- **Phase 3 – Diffusion Model (planned)**
  - Discrete diffusion / denoising model over amino acid tokens.
  - Sample valid, diverse peptide sequences under structural and property constraints.

- **Phase 4 – Property Scorers (planned)**
  - Ensemble of ML models for multiple peptide properties (e.g., hemolysis, solubility, other developability metrics).
  - Fast scoring interface to plug into generators and screening loops.

- **Phase 5 – Closed-Loop Pipeline (planned)**
  - Active learning cycles that propose, score, and (in a real setting) test new sequences.
  - Track hit rate improvements over multiple design–evaluate iterations.

- **Phase 6 – Analysis & Scaling (planned)**
  - Compare model families (classifier, LM, diffusion, ensemble).
  - Perform motif analysis, interpretability, and plan for larger-scale training.

## Current implementation status

- ✅ Phase 0: data sourcing, feature engineering, rule-based solubility heuristic, and EDA utilities (`phase0_peptide_data_prep_eda.py`).
- ✅ Phase 1: ESM-2-based hemolysis classifier with full training/evaluation loop and saved artifacts (`phase1_esm2_hemolysis_classifier.py`).
- 🔜 Phases 2–6: generative modeling, property scoring, and closed-loop optimization.

## Environment

The project uses a conda environment tailored for PyTorch, Hugging Face, and scientific Python.

To create it:

```bash
conda env create -f environment.yml
conda activate peptide-design
