"""
============================================================
🎵 HumDial – Clear MFCC & Spectrogram Visualizations
Member A Task: Audio Feature Visualization (Improved)
============================================================
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- STEP 1: Paths ---
dataset_path = r"D:\HD-Track2"
output_dir = r"D:\ALVIS\plots\MFCC_Spectrograms_Clean"
os.makedirs(output_dir, exist_ok=True)

# --- STEP 2: Collect all .wav files ---
audio_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".wav"):
            audio_files.append(os.path.join(root, file))

print(f"🎧 Found {len(audio_files)} audio files")

# --- STEP 3: Enhanced Visualization Settings ---
plt.rcParams.update({
    "axes.facecolor": "#f5f5f5",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 11,
    "figure.dpi": 150
})

# --- STEP 4: Extract and Plot Features ---
sample_limit = 10   # Adjust for testing or fewer plots

for i, file_path in enumerate(tqdm(audio_files[:sample_limit], desc="🔍 Generating Clean MFCC & Spectrograms")):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        file_name = os.path.basename(file_path)

        # --- Clean Spectrogram ---
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=512)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='log', cmap='plasma')
        plt.title(f"Spectrogram: {file_name}", fontsize=13, fontweight='bold', pad=10)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        cbar = plt.colorbar(format='%+2.0f dB')
        cbar.set_label('Intensity (dB)', rotation=270, labelpad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"spectrogram_{i+1:04d}.pdf"), bbox_inches='tight')
        plt.close()

        # --- Clean MFCC Plot ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"MFCCs and Derivatives: {file_name}", fontsize=13, fontweight='bold')

        # Base MFCCs
        img1 = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax[0], cmap='viridis')
        ax[0].set_ylabel("MFCC Coefficients")
        fig.colorbar(img1, ax=ax[0], orientation='horizontal', pad=0.1, aspect=40)

        # ΔMFCC (First derivative)
        img2 = librosa.display.specshow(delta, x_axis='time', sr=sr, ax=ax[1], cmap='coolwarm')
        ax[1].set_ylabel("ΔMFCC")
        fig.colorbar(img2, ax=ax[1], orientation='horizontal', pad=0.1, aspect=40)

        # Δ²MFCC (Second derivative)
        img3 = librosa.display.specshow(delta2, x_axis='time', sr=sr, ax=ax[2], cmap='coolwarm')
        ax[2].set_ylabel("Δ²MFCC")
        ax[2].set_xlabel("Time (s)")
        fig.colorbar(img3, ax=ax[2], orientation='horizontal', pad=0.15, aspect=40)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, f"mfcc_clean_{i+1:04d}.pdf"), bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"⚠️ Error with {file_path}: {e}")

print("\n✅ Clean MFCC and Spectrogram visualizations saved in:")
print(f"   {output_dir}")
