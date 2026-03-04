import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================================================================================
# CONFIGURATION
# ======================================================================================
CSV_PATH = "/mnt/d/ClothSim/Results/CalculatedMetrics/calculated_metrics.csv"
OUTPUT_DIR = os.path.dirname(CSV_PATH)
MODELS = ["TailorNet", "CCraft", "HOOD"]
METRICS = ["rmse", "accel_err", "collision"]

# Thresholds for visualization filtering only
RMSE_LIMIT = 0.3
ACCEL_LIMIT = 0.06

def export_histograms(df, material, suffix="original"):
    """Generates 1x3 boxplots for a specific material and cut."""
    if df.empty: return
    
    plt.figure(figsize=(18, 6))
    for i, metric in enumerate(METRICS):
        plt.subplot(1, 3, i+1)
        cols = [f"{m}_{metric}" for m in MODELS if f"{m}_{metric}" in df.columns]
        melted = df.melt(value_vars=cols, var_name='Model', value_name=metric)
        melted['Model'] = melted['Model'].str.split('_').str[0]
        
        sns.boxplot(data=melted, x='Model', y=metric, palette="husl", fliersize=3, legend=False, hue='Model')
        plt.title(f"{metric.upper()} - {material.upper()} ({suffix.upper()})")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        if (material == "cotton" and suffix == "no_outliers") or (material == "silk" and suffix == "with_outliers"):
            limits = [(0.01, 0.15), (-0.001, 0.023), (-0.25, 11)]
            plt.ylim(limits[i])

    plt.tight_layout()
    filename = f"histograms_{material}_{suffix}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def get_summary(df):
    """Calculates means for all metrics across models."""
    res = {}
    for m in MODELS:
        for met in METRICS:
            res[f"{m}_{met}"] = df[f"{m}_{met}"].mean()
    return res

def run_analysis():
    df = pd.read_csv(CSV_PATH)

    # --- TIME COMPUTATION BLOCK ---
    print("\n" + "="*63)
    print("TIME AVERAGES PER MODEL")
    print("="*63)
    plt.figure(figsize=(8, 5))
    results_root = os.path.dirname(OUTPUT_DIR)
    
    for m in ["TailorNet", "ccraft", "hood"]:
        t_path = os.path.join(results_root, m, "times.txt")
        if os.path.exists(t_path):
            # Read only the 5th column (index 4) containing the float time value
            tdf = pd.read_csv(t_path, sep=r'\s+', header=None, usecols=[4], names=['time'])
            print(f"{m:<10} {tdf['time'].mean():>8.4f} sec/it")
            sns.histplot(tdf['time'], kde=True, label=m, alpha=0.5)
            
    plt.title("Time Histograms (sec/it)")
    plt.xlabel("Seconds per iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms_time.png"))
    plt.close()
    # --------------------------------------

    # Pre-calculate Cleaned Mask (for plots only)
    # We keep all models' data for a row only if all models pass the threshold
    mask_rmse = (df[[f"{m}_rmse" for m in MODELS]] <= RMSE_LIMIT).all(axis=1)
    mask_accel = (df[[f"{m}_accel_err" for m in MODELS]] <= ACCEL_LIMIT).all(axis=1)
    df_clean = df[mask_rmse & mask_accel].copy()

    # --- SECTION 1: HISTOGRAM EXPORTS ---
    for mat in ["cotton", "silk"]:
        # Original (with outliers)
        export_histograms(df[df['material'] == mat], mat, suffix="with_outliers")
        # Cleaned (without outliers)
        export_histograms(df_clean[df_clean['material'] == mat], mat, suffix="no_outliers")

    # --- SECTION 2: PRINTED TABLES ---

    # A. Global Results (Cotton Only, Includes Outliers)
    print("\n" + "="*63)
    print("SECTION 1: GLOBAL RESULTS (COTTON ONLY, INCLUDES OUTLIERS)")
    print("="*63)
    global_cotton = df[df['material'] == 'cotton']
    s = get_summary(global_cotton)
    print(pd.DataFrame([{met: s[f"{m}_{met}"] for met in METRICS} for m in MODELS], index=MODELS))

    # B. Separate Garment Types (Cotton Only, Includes Outliers)
    print("\n" + "="*63)
    print("SECTION 2: GARMENT TYPE BREAKDOWN (COTTON ONLY)")
    print("="*63)
    garment_results = []
    for g in df['garment'].unique():
        subset = global_cotton[global_cotton['garment'] == g]
        if not subset.empty:
            s = get_summary(subset)
            s['garment'] = g
            garment_results.append(s)
    
    gar_df = pd.DataFrame(garment_results).set_index('garment')
    # Reorganize columns for readability (RMSE first for all models)
    rmse_cols = [f"{m}_rmse" for m in MODELS]
    print(gar_df[rmse_cols])

    # C. Separate Cloth Types (Cotton vs Silk, Includes Outliers)
    print("\n" + "="*63)
    print("SECTION 3: CLOTH MATERIAL COMPARISON (ALL GARMENTS)")
    print("="*63)
    material_results = []
    for m_type in ["cotton", "silk"]:
        subset = df[df['material'] == m_type]
        if not subset.empty:
            s = get_summary(subset)
            s['material'] = m_type
            material_results.append(s)
    
    mat_df = pd.DataFrame(material_results).set_index('material')
    # Focus on Acceleration and Collision for Material analysis
    accel_cols = [f"{m}_accel_err" for m in MODELS]
    col_cols = [f"{m}_collision" for m in MODELS]
    print("\nAcceleration Error by Material:")
    print(mat_df[accel_cols])
    print("\nCollision Rate (%) by Material:")
    print(mat_df[col_cols])

if __name__ == "__main__":
    run_analysis()