import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

# Paths
base_path = "data/HD-Track2/HD-Track2-dev/HD-Track2-dev-en"
output_dir = "plots/Duration and Waveform Analysis/"
os.makedirs(output_dir, exist_ok=True)

# Load durations CSV to pick sample clips
durations_csv = os.path.join("main/EDA/csv/", "durations.csv")
df = pd.read_csv(durations_csv)

# Plot sample waveforms
sample_clips = df.sample(min(3, len(df)))
for i, row in sample_clips.iterrows():
    file_path = os.path.join(base_path, row['scenario'], row['filename'])
    y, sr = librosa.load(file_path)
    plt.figure(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {row['filename']} ({row['duration_sec']:.2f}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    waveform_path = os.path.join(output_dir, f"waveform_{i}.pdf")
    plt.savefig(waveform_path, format="pdf")
    plt.show()
    print(f"Waveform saved to {waveform_path}")