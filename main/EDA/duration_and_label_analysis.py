import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

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

# --- STEP 7: Plot Charts with Better Label Spacing ---
os.makedirs("plots", exist_ok=True)

# --- Improved Bar Chart ---
plt.figure(figsize=(max(10, len(label_counts) * 0.8), 6))  # Dynamic width based on number of labels
bars = plt.bar(range(len(label_counts)), label_counts.values, 
               color='lightsteelblue', edgecolor='navy', alpha=0.8, linewidth=0.5)

plt.title("Label Distribution (Bar Chart)", fontsize=14, pad=20)
plt.xlabel("Emotion Labels", fontsize=12)
plt.ylabel("Number of Audio Files", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Improved x-axis labels with rotation and better spacing
plt.xticks(range(len(label_counts)), label_counts.index, 
           rotation=45, ha='right', rotation_mode='anchor', fontsize=10)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(label_counts.values)*0.01,
             f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Adjust y-axis limit to accommodate value labels
plt.ylim(0, max(label_counts.values) * 1.1)

plt.tight_layout()
plt.savefig("plots/Label Distribution/label_distribution_bar.pdf", format="pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- Improved Pie Chart ---
plt.figure(figsize=(12, 9))
colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))

# Create pie chart with better label positioning
wedges, texts, autotexts = plt.pie(
    label_counts.values,
    labels=None,  # We'll add custom labels later
    autopct='',   # We'll add custom percentage text
    startangle=90,
    colors=colors,
    wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'alpha': 0.9},
    textprops={'fontsize': 10}
)

# Custom legend with better formatting
legend_labels = [f'{label} ({count}, {count/sum(label_counts.values)*100:.1f}%)' 
                 for label, count in zip(label_counts.index, label_counts.values)]
plt.legend(wedges, legend_labels, 
           title="Emotion Labels",
           loc="center left",
           bbox_to_anchor=(1, 0, 0.5, 1),
           fontsize=10,
           frameon=True,
           fancybox=True,
           shadow=True)

plt.title("Label Distribution (Pie Chart)", fontsize=14, pad=20)

plt.tight_layout()
plt.savefig("plots/Label Distribution/label_distribution_pie.pdf", format="pdf", bbox_inches='tight', dpi=300)
plt.show()

# Alternative Pie Chart with direct labels (if you prefer this style)
plt.figure(figsize=(12, 8))
colors = plt.cm.tab20.colors

# For pie chart with direct labels (better for fewer categories)
if len(label_counts) <= 8:
    wedges, texts, autotexts = plt.pie(
        label_counts.values,
        labels=[f'{label}\n({count})' for label, count in zip(label_counts.index, label_counts.values)],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(label_counts)],
        textprops={'fontsize': 9, 'ha': 'center'},
        labeldistance=1.1,
        pctdistance=0.85
    )
    
    # Improve percentage text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
else:
    # For many categories, use the legend approach
    wedges, texts, autotexts = plt.pie(
        label_counts.values,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(label_counts)],
        textprops={'fontsize': 8}
    )

plt.title("Label Distribution (Pie Chart - Direct Labels)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("plots/Label Distribution/label_distribution_pie_direct.pdf", format="pdf", bbox_inches='tight', dpi=300)
plt.show()

# --- STEP 8: Save summaries ---
summary = {
    "Total Audio Files": [len(audio_files)],
    "Total Duration (Seconds)": [round(total_duration, 2)],
    "Total Duration (Hours)": [round(total_duration / 3600, 2)],
    "Unique Labels": [len(label_counts)]
}

df_summary = pd.DataFrame(summary)
df_summary.to_csv("main/EDA/csv/dataset_summary.csv", index=False)
df.to_csv("main/EDA/csv/file_labels.csv", index=False)

print("\n✅ Summary saved:")
print(" - dataset_summary.csv (overview)")
print(" - file_labels.csv (each file + label)")
print(" - plots/label_distribution_bar.pdf")
print(" - plots/label_distribution_pie.pdf")
print(" - plots/label_distribution_pie_direct.pdf (alternative)")

# Additional: Print some statistics
print(f"\n📊 Dataset Statistics:")
print(f"   - Number of unique labels: {len(label_counts)}")
print(f"   - Most common label: {label_counts.index[0]} ({label_counts.values[0]} files)")
print(f"   - Least common label: {label_counts.index[-1]} ({label_counts.values[-1]} files)")
print(f"   - Average files per label: {len(audio_files)/len(label_counts):.1f}")