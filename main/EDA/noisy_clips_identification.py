import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reuse analyze_dataset and DATA_ROOT from existing analysis script
from noise_analysis import analyze_dataset, DATA_ROOT

def build_feature_correlation_heatmap(root: Path, out_prefix="hdtrack2"):
    # compute per-file metrics (will also save hdtrack2_noise_metrics.csv via analyze_dataset)
    df = analyze_dataset(root)

    # select numeric columns relevant to noise
    numeric_cols = ["mean_rms_db", "median_rms_db", "std_rms_db", "silence_ratio", "num_frames"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df_num = df[numeric_cols].copy()

    # drop rows with all-NaN numeric values
    df_num = df_num.dropna(how="all")

    if df_num.empty:
        print("No numeric features available to correlate.")
        return df, None

    # compute correlation matrix
    corr = df_num.corr(method="pearson")

    # plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": .8})
    plt.title("Feature correlation heatmap (per-file noise metrics)")
    plt.tight_layout()
    out_png = f"{out_prefix}_feature_correlation_heatmap.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved feature correlation heatmap to {out_png}")

    return df, corr

def identify_top_noisy_clips(df: pd.DataFrame, top_n=50):
    # define a simple noise score: lower mean RMS + higher silence ratio => higher noise score
    # normalize each metric to 0..1 across dataset before combining
    metrics = {}
    if "mean_rms_db" in df.columns:
        metrics["mean_rms_db"] = (df["mean_rms_db"].max() - df["mean_rms_db"])  # invert: lower mean -> higher
    if "silence_ratio" in df.columns:
        metrics["silence_ratio"] = df["silence_ratio"]
    if not metrics:
        print("No metrics available to rank noisy clips.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics)
    # min-max normalize
    metrics_norm = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min() + 1e-12)
    metrics_norm["noise_score"] = metrics_norm.sum(axis=1)
    ranked = df.assign(noise_score=metrics_norm["noise_score"]).sort_values("noise_score", ascending=False)
    out_csv = "hdtrack2_top_noisy_clips.csv"
    ranked.head(top_n).to_csv(out_csv, index=False)
    print(f"Saved top {top_n} noisy clips to {out_csv}")
    return ranked

if __name__ == "__main__":
    print(f"Scanning dataset root: {DATA_ROOT}")
    df, corr = build_feature_correlation_heatmap(DATA_ROOT)
    ranked = identify_top_noisy_clips(df, top_n=100)
    # print brief summary
    print(ranked[["file", "mean_rms_db", "silence_ratio", "noise_score"]].head(10))