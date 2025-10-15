import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt

# Paths
base_path = "data/HD-Track2/HD-Track2-dev/HD-Track2-dev-en"
output_dir = "plots/Duration and Waveform Analysis/"
os.makedirs(output_dir, exist_ok=True)

# Measure durations
records = []

for scenario in os.listdir(base_path):
    scenario_path = os.path.join(base_path, scenario)
    if os.path.isdir(scenario_path):
        for file in os.listdir(scenario_path):
            if file.endswith(".wav"):
                file_path = os.path.join(scenario_path, file)
                try:
                    duration = librosa.get_duration(filename=file_path)
                    records.append({
                        "scenario": scenario,
                        "filename": file,
                        "duration_sec": duration
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Save durations CSV
df = pd.DataFrame(records)
durations_csv = os.path.join(output_dir, "durations.csv")
df.to_csv(durations_csv, index=False)
print(f"Durations saved to {durations_csv}")

# Duration histogram
plt.figure(figsize=(8,5))
plt.hist(df["duration_sec"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of clips")
plt.title("Clip Duration Distribution")
plt.tight_layout()
hist_path = os.path.join(output_dir, "duration_histogram.pdf")
plt.savefig(hist_path)
plt.show()
print(f"Histogram saved to {hist_path}")