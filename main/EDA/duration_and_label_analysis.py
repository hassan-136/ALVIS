
import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- STEP 1: Set Dataset Path ---
dataset_path = r"D:\HD-Track2"

# --- STEP 2: Collect all .wav file paths ---
audio_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".wav"):
            audio_files.append(os.path.join(root, file))

print(f"🎧 Total audio files found: {len(audio_files)}")

# --- STEP 3: Calculate total duration ---
total_duration = 0.0
print("\n⏳ Calculating total duration...")
for file in tqdm(audio_files):
    try:
        y, sr = librosa.load(file, sr=None)
        total_duration += librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"⚠️ Skipped {file} ({e})")

hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)
print(f"\n🕒 Total dataset duration: {hours}h {minutes}m {seconds}s")

# --- STEP 4: Extract labels automatically ---
labels = []
for path in audio_files:
    filename = os.path.basename(path)
    parts = filename.split('_')
    if len(parts) > 1:
        label = parts[0].lower()
    else:
        label = os.path.basename(os.path.dirname(path)).lower()
    labels.append(label)

# --- STEP 5: Create DataFrame ---
df = pd.DataFrame({
    "file_path": audio_files,
    "label": labels
})

# --- STEP 6: Label Distribution ---
label_counts = df["label"].value_counts()
print("\n🏷️ Label Distribution:")
print(label_counts)

# --- STEP 7: Plot Charts and Save as PDF ---
os.makedirs("plots", exist_ok=True)

# --- Bar Chart ---
plt.figure(figsize=(8,5))
label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Label Distribution (Bar Chart)")
plt.xlabel("Emotion Labels")
plt.ylabel("Number of Audio Files")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/label_distribution_bar.pdf", format="pdf")
plt.show()

# --- Pie Chart (cleaner labels & colors) ---
plt.figure(figsize=(7,7))
colors = plt.cm.tab20.colors  # better contrast
plt.pie(
    label_counts,
    labels=[f"{label} ({count})" for label, count in zip(label_counts.index, label_counts.values)],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors[:len(label_counts)],
    textprops={'fontsize': 10}
)
plt.title("Label Distribution (Pie Chart)")
plt.tight_layout()
plt.savefig("plots/label_distribution_pie.pdf", format="pdf")
plt.show()

# --- STEP 8: Save summaries ---
summary = {
    "Total Audio Files": [len(audio_files)],
    "Total Duration (Seconds)": [round(total_duration, 2)],
    "Total Duration (Hours)": [round(total_duration / 3600, 2)],
    "Unique Labels": [len(label_counts)]
}

df_summary = pd.DataFrame(summary)
df_summary.to_csv("dataset_summary.csv", index=False)
df.to_csv("file_labels.csv", index=False)

print("\n✅ Summary saved:")
print(" - dataset_summary.csv (overview)")
print(" - file_labels.csv (each file + label)")
print(" - plots/label_distribution_bar.pdf")
print(" - plots/label_distribution_pie.pdf")
