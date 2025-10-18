import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_ROOT = Path(__file__).parent / "HD-Track2" / "HD-Track2"  # change if your data root differs
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SILENCE_DB_THRESHOLD = -40.0  # frames with RMS (dB) below this are considered silence
MAX_FRAMES_TO_SAMPLE = 2000000  # cap total frames aggregated for frame-level histogram


def find_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            yield p


def rms_db_from_audio(path: Path, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if y.size == 0:
        return np.array([]), sr
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # convert amplitude RMS to dB
    rms_db = 20.0 * np.log10(rms + 1e-10)
    return rms_db, sr


def analyze_dataset(root: Path):
    rows = []
    all_frame_rms = []
    total_frames = 0

    audio_files = list(find_audio_files(root))
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {root}")

    for audio_path in tqdm(audio_files, desc="Analyzing audio files"):
        try:
            rms_db, sr = rms_db_from_audio(audio_path)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            continue

        if rms_db.size == 0:
            rows.append({
                "file": str(audio_path.relative_to(root)),
                "mean_rms_db": np.nan,
                "median_rms_db": np.nan,
                "std_rms_db": np.nan,
                "silence_ratio": np.nan,
                "num_frames": 0
            })
            continue

        mean_db = float(np.mean(rms_db))
        median_db = float(np.median(rms_db))
        std_db = float(np.std(rms_db))
        silence_ratio = float(np.mean(rms_db < SILENCE_DB_THRESHOLD))
        num_frames = int(rms_db.size)

        rows.append({
            "file": str(audio_path.relative_to(root)),
            "mean_rms_db": mean_db,
            "median_rms_db": median_db,
            "std_rms_db": std_db,
            "silence_ratio": silence_ratio,
            "num_frames": num_frames,
            "sr": sr
        })

        # aggregate frame-level RMS for global histogram (cap total frames)
        if total_frames < MAX_FRAMES_TO_SAMPLE:
            remaining = MAX_FRAMES_TO_SAMPLE - total_frames
            take = min(remaining, rms_db.size)
            all_frame_rms.append(rms_db[:take])
            total_frames += take

    df = pd.DataFrame(rows)
    all_frame_rms = np.concatenate(all_frame_rms) if all_frame_rms else np.array([])

    # Save per-file metrics
    out_csv = Path.cwd() / "hdtrack2_noise_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved per-file metrics to {out_csv}")

    # Plot histograms
    fig1, ax1 = plt.subplots()
    # histogram of per-file mean RMS (dB)
    ax1.hist(df["mean_rms_db"].dropna(), bins=50, color="C0", alpha=0.8)
    ax1.set_title("Histogram of per-file mean RMS (dB)")
    ax1.set_xlabel("Mean RMS (dB)")
    ax1.set_ylabel("Number of files")
    fig1.tight_layout()
    fig1.savefig("hist_mean_rms_per_file.png", dpi=150)
    print("Saved hist_mean_rms_per_file.png")

    fig2, ax2 = plt.subplots()
    # histogram of per-file silence ratio
    ax2.hist(df["silence_ratio"].dropna(), bins=50, color="C1", alpha=0.8)
    ax2.set_title(f"Histogram of per-file silence ratio (threshold {SILENCE_DB_THRESHOLD} dB)")
    ax2.set_xlabel("Silence ratio")
    ax2.set_ylabel("Number of files")
    fig2.tight_layout()
    fig2.savefig("hist_silence_ratio_per_file.png", dpi=150)
    print("Saved hist_silence_ratio_per_file.png")

    # frame-level RMS histogram (aggregated sample)
    if all_frame_rms.size:
        fig3, ax3 = plt.subplots()
        ax3.hist(all_frame_rms, bins=100, color="C2", alpha=0.7)
        ax3.set_title("Aggregated frame-level RMS (dB) across dataset (sampled)")
        ax3.set_xlabel("RMS (dB)")
        ax3.set_ylabel("Frames (sampled)")
        fig3.tight_layout()
        fig3.savefig("hist_frame_rms_sampled.png", dpi=150)
        print("Saved hist_frame_rms_sampled.png")

    return df


if __name__ == "__main__":
    print(f"Scanning dataset root: {DATA_ROOT}")
    df = analyze_dataset(DATA_ROOT)
    print(df.describe(include="all"))

