"""
Phase 0: Peptide Dataset Sourcing and EDA
==========================================

This script downloads open-source peptide datasets, preprocesses them,
creates sampled subsets for CPU-friendly development, and performs exploratory
data analysis.

Datasets sourced:
1. Hemolysis (from AMPDeep / AMPDeep-like sources)
2. Rule-based solubility scores computed on hemolysis sequences
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Bio import SeqIO
import requests
from typing import List, Tuple, Dict
import urllib.request
import zipfile

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path('./data')
DATA_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Amino acid properties (for feature extraction)
AA_CHARGE = {
    'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1,  # ionizable
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0,
    'V': 0, 'W': 0, 'Y': 0
}

AA_HYDROPHOBICITY = {
    # Kyte-Doolittle scale
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# ============================================================================
# 1. Download and Prepare Hemolysis Dataset (AMPDeep)
# ============================================================================

def download_ampdeep_hemolysis() -> pd.DataFrame:
    """
    Download AMPDeep hemolysis dataset from supplementary materials.
    
    This dataset contains antimicrobial peptides with hemolysis labels.
    Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9511757/
    GitHub: https://github.com/milad73s/AMPDeep/tree/main
    
    Returns:
        DataFrame with columns: sequence, hemolytic (0/1)
    """
    print("\n[HEMOLYSIS DATA] Downloading AMPDeep hemolysis dataset...")
    
    # Since direct download link may vary, we'll create a representative dataset
    # from known hemolysis properties. In practice, you'd download from:
    # https://github.com/PeptoneLtd/AMPDeep/tree/master/datasets
    
    # For this demo, create a synthetic hemolysis dataset based on known patterns:
    # - Highly charged peptides tend to be hemolytic
    # - Hydrophobic peptides tend to be hemolytic
    # - Length matters
    
    hemolysis_data = {
        'sequence': [
            'DWFKAFYDKVAEKDLYDKLWSDLYDKL',
            'GFFO',
            'KRVMRGK',
            'GIGAVLKVLTTV',
            'MNMQLLTSK',
            'VSHDVAKF',
        ] * 100,  # Repeat for ~600 sequences (we'll sample down)
        'hemolytic': [1, 0, 1, 0, 0, 1] * 100,
    }
    
    df_hemolysis = pd.DataFrame(hemolysis_data)
    
    # Try to download real data if available
    try:
        url = "https://raw.githubusercontent.com/milad73s/AMPDeep/main/data/hemolytic/hemolythic.csv"
        df_hemolysis_online = pd.read_csv(url, usecols=['text', 'labels', 'split'])
        df_hemolysis_online.columns = ['sequence', 'hemolytic', 'split']
        df_hemolysis_online['sequence'] = df_hemolysis_online['sequence'].str.upper()
        df_hemolysis_online['hemolytic'] = df_hemolysis_online['hemolytic'].astype(int)
        
        print(f"  Downloaded {len(df_hemolysis_online)} sequences from online source")
        return df_hemolysis_online
    except Exception as e:
        print(f"  Could not download online source ({e}). Using demo data.")
        print(f"  → Proceeding with synthetic hemolysis data ({len(df_hemolysis)} sequences)")
        return df_hemolysis

# ============================================================================
# 3. Feature Extraction
# ============================================================================

def compute_peptide_features(seq: str) -> Dict[str, float]:
    """Compute physicochemical features for a peptide sequence."""
    if len(seq) == 0:
        return {
            'length': 0,
            'charge': 0.0,
            'hydrophobicity': 0.0,
            'polar_fraction': 0.0,
        }
    
    charge = sum(AA_CHARGE.get(aa, 0) for aa in seq) / len(seq)
    hydro = sum(AA_HYDROPHOBICITY.get(aa, 0) for aa in seq) / len(seq)
    polar_count = sum(1 for aa in seq if aa in {'D', 'E', 'K', 'R', 'S', 'T', 'N', 'Q', 'H'})
    polar_frac = polar_count / len(seq)
    
    return {
        'length': len(seq),
        'charge': charge,
        'hydrophobicity': hydro,
        'polar_fraction': polar_frac,
    }


def add_features_to_dataframe(df: pd.DataFrame, seq_col: str = 'sequence') -> pd.DataFrame:
    """Add computed features to DataFrame."""
    features = [compute_peptide_features(seq) for seq in df[seq_col]]
    features_df = pd.DataFrame(features)
    return pd.concat([df, features_df], axis=1)

def add_rule_based_solubility(df: pd.DataFrame,
                              length_col: str = "length",
                              charge_col: str = "charge",
                              hydro_col: str = "hydrophobicity",
                              polar_col: str = "polar_fraction",
                              score_col: str = "solubility_score",
                              label_col: str = "soluble_rule") -> pd.DataFrame:
    """
    Add a heuristic solubility score and binary label based on simple rules:
    - More polar → more soluble
    - Less hydrophobic → more soluble
    - Moderate length → more soluble
    - Extreme positive charge → slightly less soluble (aggregation/interaction risk)
    """
    L = df[length_col].astype(float)
    Q = df[charge_col].astype(float)
    H = df[hydro_col].astype(float)
    P = df[polar_col].astype(float)

    # Normalize/clip to reasonable ranges for stability
    L_clipped = L.clip(lower=5, upper=50)
    H_clipped = H.clip(lower=-3.0, upper=3.0)
    Q_abs = Q.abs().clip(upper=2.0)

    # Base score: start around 0.0
    score = np.zeros_like(P, dtype=float)

    # Polar fraction: strong positive effect
    score += 2.0 * (P - 0.3)          # >0.3 helps, <0.3 hurts

    # Hydrophobicity: more hydrophobic (higher H) → less soluble
    score += -1.5 * (H_clipped - 0.0)

    # Length: penalize very long peptides, slight bonus for moderate length
    score += -0.03 * (L_clipped - 20.0)  # >20 aa mildly penalized, <20 mildly boosted

    # Charge: very high |charge| can reduce solubility
    score += -0.8 * (Q_abs - 0.5)

    # Map to 0–1 with logistic
    sol_prob = 1.0 / (1.0 + np.exp(-score))

    df[score_col] = sol_prob
    df[label_col] = (sol_prob >= 0.5).astype(int)
    return df

# ============================================================================
# 4. Data Preparation
# ============================================================================

def load_and_prepare_data() -> pd.DataFrame:
    """
    Load or create the hemolysis dataset and add features + rule-based solubility.

    Returns:
        hemolysis_df
    """
    df_hemolysis = download_ampdeep_hemolysis()
    df_hemolysis = add_features_to_dataframe(df_hemolysis, 'sequence')
    df_hemolysis = add_rule_based_solubility(df_hemolysis)
    return df_hemolysis

# ============================================================================
# 5. Sampling and Export
# ============================================================================

def create_sample_dataset(
    df_hemolysis: pd.DataFrame,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """Create and export a sampled hemolysis dataset for CPU-friendly development."""
    print(f"\n[SAMPLING] Creating {sample_size}-sequence hemolysis subset...")

    if len(df_hemolysis) < sample_size:
        print(f" WARNING: Hemolysis has {len(df_hemolysis)} < {sample_size}, using all")
        sample_hem = df_hemolysis.copy()
    else:
        sample_hem = df_hemolysis.sample(
            n=sample_size, random_state=RANDOM_SEED, replace=False
        )

    hem_path = DATA_DIR / f"amp_hemolysis_sample_{len(sample_hem)}.csv"
    sample_hem.to_csv(hem_path, index=False)
    print(f" Saved: {hem_path}")

    return sample_hem

# ============================================================================
# 6. Exploratory Data Analysis
# ============================================================================

def eda_hemolysis(df: pd.DataFrame) -> None:
    """EDA for hemolysis dataset."""
    
    print("\n" + "="*70)
    print("HEMOLYSIS DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nLabel distribution (hemolytic):")
    print(df['hemolytic'].value_counts())
    print(f"Label balance: {df['hemolytic'].value_counts(normalize=True)}")
    
    print(f"\nSequence statistics:")
    print(f"  Length: {df['length'].describe()}")
    print(f"  Charge: {df['charge'].describe()}")
    print(f"  Hydrophobicity: {df['hydrophobicity'].describe()}")
    print(f"  Polar fraction: {df['polar_fraction'].describe()}")
    
    print(f"\nProperty differences by label:")
    grouped = df.groupby('hemolytic')[['length', 'charge', 'hydrophobicity', 'polar_fraction']].mean()
    print(grouped)
    
    print(f"\nAA frequency (overall):")
    all_aas = ''.join(df['sequence'])
    aa_freq = pd.Series(list(all_aas)).value_counts().sort_index()
    print(aa_freq)


def eda_solubility(df: pd.DataFrame) -> None:
    """EDA for solubility dataset."""
    
    print("\n" + "="*70)
    print("SOLUBILITY DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nLabel distribution (soluble):")
    print(df['soluble_rule'].value_counts())
    print(f"Label balance: {df['soluble_rule'].value_counts(normalize=True)}")
    
    print(f"\nSequence statistics:")
    print(f"  Length: {df['length'].describe()}")
    print(f"  Charge: {df['charge'].describe()}")
    print(f"  Hydrophobicity: {df['hydrophobicity'].describe()}")
    print(f"  Polar fraction: {df['polar_fraction'].describe()}")
    
    print(f"\nProperty differences by label:")
    grouped = df.groupby('soluble_rule')[['length', 'charge', 'hydrophobicity', 'polar_fraction']].mean()
    print(grouped)


def create_eda_plots(df_hem: pd.DataFrame) -> None:
    """Create and save EDA plots for hemolysis dataset."""
    plots_dir = DATA_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # ============================================================================
    # Figure 1: Hemolysis - Length and Charge Distribution
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hemolysis Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # Length by label
    ax = axes[0, 0]
    df_hem.boxplot(column='length', by='hemolytic', ax=ax)
    ax.set_xlabel('Hemolytic (0=No, 1=Yes)')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Length Distribution by Label')
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    # Charge by label
    ax = axes[0, 1]
    df_hem.boxplot(column='charge', by='hemolytic', ax=ax)
    ax.set_xlabel('Hemolytic (0=No, 1=Yes)')
    ax.set_ylabel('Average Charge')
    ax.set_title('Charge Distribution by Label')
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    # Hydrophobicity by label
    ax = axes[1, 0]
    df_hem.boxplot(column='hydrophobicity', by='hemolytic', ax=ax)
    ax.set_xlabel('Hemolytic (0=No, 1=Yes)')
    ax.set_ylabel('Average Hydrophobicity')
    ax.set_title('Hydrophobicity Distribution by Label')
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    # Label balance
    ax = axes[1, 1]
    df_hem['hemolytic'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_xlabel('Hemolytic')
    ax.set_ylabel('Count')
    ax.set_title('Label Balance')
    ax.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hemolysis_eda.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved: {plots_dir / 'hemolysis_eda.png'}")
    plt.close()
    
    # ============================================================================
    # Figure 2: Solubility - Properties by Label
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Solubility Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    df_hem.boxplot(column='length', by='soluble_rule', ax=ax)
    ax.set_xlabel('Soluble (0=No, 1=Yes)')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Length Distribution by Label')
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    ax = axes[0, 1]
    df_hem.boxplot(column='polar_fraction', by='soluble_rule', ax=ax)
    ax.set_xlabel('Soluble (0=No, 1=Yes)')
    ax.set_ylabel('Polar Fraction')
    ax.set_title('Polar Fraction Distribution by Label')
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    ax = axes[1, 0]
    df_hem['soluble_rule'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_xlabel('Soluble')
    ax.set_ylabel('Count')
    ax.set_title('Label Balance')
    ax.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
    
    ax = axes[1, 1]
    df_hem.boxplot(column='charge', by='soluble_rule', ax=ax)
    ax.set_xlabel('Soluble (0=No, 1=Yes)')
    ax.set_ylabel('Average Charge')
    ax.set_title('Charge Distribution by Label')
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'solubility_eda.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'solubility_eda.png'}")
    plt.close()
    
    # ============================================================================
    # Figure 3: Correlation Heatmap (Hemolysis)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_hem[['length', 'charge', 'hydrophobicity', 'polar_fraction', 'hemolytic']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Hemolysis Dataset - Feature Correlation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'hemolysis_correlation.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'hemolysis_correlation.png'}")
    plt.close()
    
    # ============================================================================
    # Figure 4: Scatter - Charge vs Hydrophobicity
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    for label, color in zip([0, 1], ['blue', 'red']):
        mask = df_hem['hemolytic'] == label
        ax.scatter(df_hem[mask]['charge'], df_hem[mask]['hydrophobicity'], 
                  label=f"Hemolytic={label}", alpha=0.5, s=30, color=color)
    ax.set_xlabel('Average Charge')
    ax.set_ylabel('Average Hydrophobicity')
    ax.set_title('Hemolysis: Charge vs Hydrophobicity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for label, color in zip([0, 1], ['blue', 'green']):
        mask = df_hem['soluble_rule'] == label
        ax.scatter(df_hem[mask]['charge'], df_hem[mask]['polar_fraction'],
                  label=f"Soluble={label}", alpha=0.5, s=30, color=color)
    ax.set_xlabel('Average Charge')
    ax.set_ylabel('Polar Fraction')
    ax.set_title('Solubility: Charge vs Polar Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'property_scatter.png', dpi=100, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'property_scatter.png'}")
    plt.close()


# ============================================================================
# 7. Create Data Manifest
# ============================================================================

def create_manifest(
    df_hem: pd.DataFrame,
) -> None:
    """Create a manifest describing the hemolysis dataset."""
    manifest = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "datasets": {
            "amp_hemolysis": {
                "filename": f"amp_hemolysis_sample_{len(df_hem)}.csv",
                "n_samples": len(df_hem),
                "columns": list(df_hem.columns),
                "label_column": "hemolytic",
                "label_description": "Binary: 1=hemolytic, 0=non-hemolytic",
                "source": "AMPDeep dataset (synthetic fallback demo possible)",
                "task": "Predict hemolytic activity",
                "label_balance": {
                    "0": int(df_hem["hemolytic"].value_counts().get(0, 0)),
                    "1": int(df_hem["hemolytic"].value_counts().get(1, 0)),
                },
            },
        },
        "amino_acid_vocabulary": list(AA_CHARGE.keys()),
        "computed_features": [
            "length",
            "charge",
            "hydrophobicity",
            "polar_fraction",
            "solubility_score",
            "soluble_rule",
        ],
        "feature_descriptions": {
            "length": "Number of amino acids in the sequence",
            "charge": "Average charge per residue (sum of AA charges / length)",
            "hydrophobicity": "Average Kyte-Doolittle hydrophobicity score",
            "polar_fraction": "Fraction of polar residues (D, E, K, R, S, T, N, Q, H)",
            "solubility_score": "Rule-based solubility score in [0,1]",
            "soluble_rule": "Binary: 1=predicted soluble (rule-based), 0=insoluble",
        },
    }

    manifest_path = DATA_DIR / "data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {manifest_path}")

# ============================================================================
# 8. Main Execution
# ============================================================================

def main():
    """Main Phase 0 workflow."""
    print("\n" + "=" * 70)
    print("PHASE 0: PEPTIDE DATASET SOURCING AND EDA")
    print("=" * 70)

    # Load/create hemolysis data with features + rule-based solubility
    df_hemolysis = load_and_prepare_data()

    # Create sampled subset for CPU-friendly development
    df_hem_sample = create_sample_dataset(df_hemolysis, sample_size=1000)

    # Print summary stats
    eda_hemolysis(df_hem_sample)
    eda_solubility(df_hem_sample)
    # Create visualizations
    print("\n[PLOTS] Creating exploratory data analysis visualizations...")
    create_eda_plots(df_hem_sample)

    # Create manifest
    create_manifest(df_hem_sample)

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print(" 1. Review plots in ./data/plots/")
    print(" 2. Check data manifest at ./data/data_manifest.json")
    print(" 3. Run Phase 1: Fine-tune ESM-2 for hemolysis prediction")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
