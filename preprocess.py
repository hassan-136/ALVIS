"""
preprocess.py  —  HumDial HD-Track2 (multilingual, shared labels)

Each category folder name (e.g. "Follow-up Questions") becomes a class label.
Both English and Chinese data are merged under the same label.
Produces:
    preprocessed/
        train/<label>/*.npy
        test/<label>/*.npy
        train_manifest.csv
        test_manifest.csv
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import csv

LANG_FOLDERS = [
    ("train", [
        "HD-Track2-train/HD-Track2-train-en",
        "HD-Track2-train/HD-Track2-train-zh"
    ]),
    ("test", [
        "HD-Track2-dev/HD-Track2-dev-en",
        "HD-Track2-dev/HD-Track2-dev-zh"
    ]),
]

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def process_file(path, sr, n_fft, hop_length, win_length, top_db=40):
    """Load → trim → normalize → log-STFT"""
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio.astype(float), orig_sr, sr)
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(stft)
    log_mag = np.log1p(mag)
    mean, std = log_mag.mean(), log_mag.std() or 1.0
    return ((log_mag - mean) / std).astype(np.float32)


def collect_and_save(data_root, out_dir, sr, n_fft, hop_length, win_length):
    manifests = {"train": [], "test": []}

    for split, folders in LANG_FOLDERS:
        for folder_name in folders:
            folder_path = os.path.join(data_root, folder_name)
            if not os.path.isdir(folder_path):
                print(f"⚠️  Missing folder {folder_path}, skipping.")
                continue

            for category in sorted(os.listdir(folder_path)):
                cat_path = os.path.join(folder_path, category)
                if not os.path.isdir(cat_path):
                    continue

                for fname in os.listdir(cat_path):
                    if not fname.lower().endswith(AUDIO_EXTS):
                        continue
                    src = os.path.join(cat_path, fname)
                    label = category  # shared label across languages

                    try:
                        spec = process_file(src, sr, n_fft, hop_length, win_length)
                    except Exception as e:
                        print(f"Failed {src}: {e}")
                        continue

                    out_sub = os.path.join(out_dir, split, label)
                    ensure_dir(out_sub)
                    out_file = os.path.join(out_sub, os.path.splitext(fname)[0] + ".npy")
                    np.save(out_file, spec)
                    manifests[split].append((out_file, label))

    # write manifests
    for split in ["train", "test"]:
        ensure_dir(out_dir)
        csv_path = os.path.join(out_dir, f"{split}_manifest.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])
            writer.writerows(manifests[split])
        print(f"✅ Wrote {csv_path} ({len(manifests[split])} samples)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Root containing HD-Track2 folders")
    p.add_argument("--out_dir", default="preprocessed", help="Output folder in ALVIS/")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--win_length", type=int, default=512)
    args = p.parse_args()

    ensure_dir(args.out_dir)
    collect_and_save(args.data_root, args.out_dir, args.sr, args.n_fft, args.hop_length, args.win_length)
    print("🎯 Preprocessing complete.")


if __name__ == "__main__":
    main()