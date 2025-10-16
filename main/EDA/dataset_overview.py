import os
import pandas as pd

# --- STEP 1: Set Dataset Path ---
# 🔹 Give the TOPMOST folder that contains everything
dataset_path = r"D:\HD-Track2"   # adjust if needed

print("Checking if path exists:", os.path.exists(dataset_path))
print("Scanning dataset folder...")

# --- STEP 2: Initialize Lists ---
audio_files = []
speaker_folders = []
dialogue_sessions = set()

# --- STEP 3: Deep Scan for WAV Files ---
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".wav"):
            full_path = os.path.join(root, file)
            audio_files.append(full_path)

            # Use folder names dynamically — even if deeply nested
            path_parts = os.path.normpath(full_path).split(os.sep)

            # Try to grab speaker folder (3rd last folder, if exists)
            if len(path_parts) >= 3:
                speaker_folders.append(path_parts[-3])

            # Try to grab dialogue session folder (4th last, if exists)
            if len(path_parts) >= 4:
                dialogue_sessions.add(path_parts[-4])

# --- STEP 4: Compute Stats ---
total_files = len(audio_files)
total_speakers = len(set(speaker_folders))
total_sessions = len(dialogue_sessions)

print("\n Dataset Overview")
print(f"Total audio files: {total_files}")
print(f"Total speakers (approx): {total_speakers}")
print(f"Total dialogue sessions (approx): {total_sessions}")

# --- STEP 5: Save Summary ---
summary_data = {
    "Total Audio Files": [total_files],
    "Estimated Speakers": [total_speakers],
    "Estimated Dialogue Sessions": [total_sessions]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("dataset_overview.csv", index=False)

print("\nDataset overview saved as 'dataset_overview.csv'")
