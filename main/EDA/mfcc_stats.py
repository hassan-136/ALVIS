"""
============================================================
📈 HumDial – MFCC Statistical Summary
Member A Task: Feature Distribution Analysis
============================================================
"""

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- STEP 1: Dataset Path ---
dataset_path = r"D:\HD-Track2"
csv_output_path = r"D:\ALVIS\EDA\csv\mfcc_stats.csv"
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

# --- STEP 2: Collect all WAV Files ---
audio_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".wav"):
            audio_files.append(os.path.join(root, file))

print(f"🎧 Total audio files: {len(audio_files)}")

# --- STEP 3: Compute MFCC Statistics ---
stats = []

for file_path in tqdm(audio_files[:300], desc="Extracting MFCC stats"):  # limit to 300 for speed
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        data = {"file": os.path.basename(file_path)}
        for i in range(13):
            data[f"mfcc{i+1}_mean"] = mfcc_mean[i]
            data[f"mfcc{i+1}_std"] = mfcc_std[i]
        stats.append(data)
    except Exception as e:
        print(f"⚠️ Skipped {file_path}: {e}")

# --- STEP 4: Save to CSV ---
df = pd.DataFrame(stats)
df.to_csv(csv_output_path, index=False)
print(f"\n✅ MFCC statistics saved to: {csv_output_path}")
