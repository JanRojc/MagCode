import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================================================
# CONFIGURATION
# ======================================================================================
CSV_PATH = "/mnt/d/ClothSim/Results/CalculatedMetrics/calculated_metrics.csv"
OUTPUT_DIR = os.path.dirname(CSV_PATH)
MODELS = ["TailorNet", "CCraft", "HOOD"]
METRICS = ["rmse", "accel_err", "collision"]
YLIM_MAX = {
    "cotton": {"rmse": 0.12, "accel_err": 0.01, "collision": 11},
    "silk": {"rmse": 0.12, "accel_err": 0.01, "collision": 11},
}
MODEL_COLORS = {
    "TailorNet": "#1f77b4",
    "CCraft": "#ff7f0e",
    "HOOD": "#2ca02c",
}

def export_histograms(df, material, suffix="original"):
    """Generates 1x3 boxplots for a specific material and cut."""
    if df.empty: return
    
    plt.figure(figsize=(18, 6))
    for i, metric in enumerate(METRICS):
        plt.subplot(1, 3, i+1)
        cols = [f"{m}_{metric}" for m in MODELS if f"{m}_{metric}" in df.columns]
        melted = df.melt(value_vars=cols, var_name='Model', value_name=metric)
        melted['Model'] = melted['Model'].str.split('_').str[0]

        data = [melted.loc[melted['Model'] == model, metric].dropna().to_numpy() for model in MODELS]
        bp = plt.boxplot(data, tick_labels=MODELS, patch_artist=True, flierprops={"markersize": 3})
        for patch, model in zip(bp["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.8)

        plt.title(f"{metric.upper()} - {material.upper()} ({suffix.upper()})")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        if suffix == "no_outliers" and material in YLIM_MAX and metric in YLIM_MAX[material]:
            plt.ylim(0, YLIM_MAX[material][metric])

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

def get_summary_std(df):
    """Calculates standard deviations for all metrics across models."""
    res = {}
    for m in MODELS:
        for met in METRICS:
            res[f"{m}_{met}"] = df[f"{m}_{met}"].std(ddof=0)
    return res

def format_mean_std(mean_val, std_val):
    return f"{mean_val:.6f} +/- {std_val:.6f}"

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
            plt.hist(tdf['time'], bins=30, alpha=0.5, label=m)
            
    plt.title("Time Histograms (sec/it)")
    plt.xlabel("Seconds per iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms_time.png"))
    plt.close()
    # --------------------------------------

    # --- SECTION 1: HISTOGRAM EXPORTS ---
    for mat in ["cotton", "silk"]:
        # Original (with outliers)
        export_histograms(df[df['material'] == mat], mat, suffix="with_outliers")
        # Full data with clipped y-axis for easier visualization
        export_histograms(df[df['material'] == mat], mat, suffix="no_outliers")

    # --- SECTION 2: PRINTED TABLES ---

    # A. Global Results (Cotton Only, Includes Outliers)
    print("\n" + "="*63)
    print("SECTION 1: GLOBAL RESULTS (COTTON ONLY, INCLUDES OUTLIERS)")
    print("="*63)
    global_cotton = df[df['material'] == 'cotton']
    s_mean = get_summary(global_cotton)
    s_std = get_summary_std(global_cotton)
    print(pd.DataFrame([{
        met: format_mean_std(s_mean[f"{m}_{met}"], s_std[f"{m}_{met}"])
        for met in METRICS
    } for m in MODELS], index=MODELS))

    # B. Separate Garment Types (Cotton Only, Includes Outliers)
    print("\n" + "="*63)
    print("SECTION 2: GARMENT TYPE BREAKDOWN (COTTON ONLY)")
    print("="*63)
    garment_results = []
    for g in df['garment'].unique():
        subset = global_cotton[global_cotton['garment'] == g]
        if not subset.empty:
            s_mean = get_summary(subset)
            s_std = get_summary_std(subset)
            row = {'garment': g}
            for m in MODELS:
                for met in METRICS:
                    key = f"{m}_{met}"
                    row[key] = format_mean_std(s_mean[key], s_std[key])
            garment_results.append(row)
    
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
            s_mean = get_summary(subset)
            s_std = get_summary_std(subset)
            row = {'material': m_type}
            for m in MODELS:
                for met in METRICS:
                    key = f"{m}_{met}"
                    row[key] = format_mean_std(s_mean[key], s_std[key])
            material_results.append(row)
    
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
