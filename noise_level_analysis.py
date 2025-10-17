import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
base_path = r"C:\Users\haniy\Downloads\HD-Track2"
output_dir = "plots/Noise Analysis/"
os.makedirs(output_dir, exist_ok=True)

# Function to calculate RMS energy
def calculate_rms(y):
    """Calculate RMS energy of audio signal"""
    return np.sqrt(np.mean(y**2))

# Function to calculate silence ratio
def calculate_silence_ratio(y, threshold_db=-40):
    """
    Calculate the ratio of silent frames in audio
    threshold_db: dB threshold below which audio is considered silent
    """
    # Convert to dB
    rms_frames = librosa.feature.rms(y=y)[0]
    db_frames = librosa.amplitude_to_db(rms_frames, ref=np.max)
    
    # Count silent frames
    silent_frames = np.sum(db_frames < threshold_db)
    total_frames = len(db_frames)
    
    return silent_frames / total_frames if total_frames > 0 else 0

# Collect noise metrics
noise_data = []

# Walk through all audio files
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(('.wav', '.flac', '.mp3')):
            file_path = os.path.join(root, file)
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=None)
                
                # Calculate metrics
                rms = calculate_rms(y)
                rms_db = librosa.amplitude_to_db(np.array([rms]))[0]
                silence_ratio = calculate_silence_ratio(y, threshold_db=-40)
                
                # Get relative path for scenario
                rel_path = os.path.relpath(root, base_path)
                scenario = rel_path.split(os.sep)[0] if os.sep in rel_path else 'root'
                
                noise_data.append({
                    'filename': file,
                    'scenario': scenario,
                    'rms': rms,
                    'rms_db': rms_db,
                    'silence_ratio': silence_ratio
                })
                
                print(f"Processed: {file}")
                
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Create DataFrame
df_noise = pd.DataFrame(noise_data)

# Save to CSV
csv_path = os.path.join(output_dir, "noise_metrics.csv")
df_noise.to_csv(csv_path, index=False)
print(f"\nNoise metrics saved to {csv_path}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. RMS (linear) histogram
axes[0, 0].hist(df_noise['rms'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('RMS Energy')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of RMS Energy (Linear)')
axes[0, 0].grid(True, alpha=0.3)

# 2. RMS (dB) histogram
axes[0, 1].hist(df_noise['rms_db'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('RMS Energy (dB)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of RMS Energy (dB)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Silence ratio histogram
axes[1, 0].hist(df_noise['silence_ratio'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Silence Ratio')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Silence Ratio')
axes[1, 0].grid(True, alpha=0.3)

# 4. RMS vs Silence Ratio scatter plot
scatter = axes[1, 1].scatter(df_noise['silence_ratio'], df_noise['rms_db'], 
                            alpha=0.5, c=df_noise['rms_db'], cmap='viridis')
axes[1, 1].set_xlabel('Silence Ratio')
axes[1, 1].set_ylabel('RMS Energy (dB)')
axes[1, 1].set_title('Silence Ratio vs RMS Energy')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='RMS (dB)')

plt.tight_layout()
plot_path = os.path.join(output_dir, "noise_level_histograms.pdf")
plt.savefig(plot_path, format="pdf", dpi=300)
plt.show()
print(f"Histograms saved to {plot_path}")

# Print summary statistics
print("\n" + "="*50)
print("NOISE METRICS SUMMARY")
print("="*50)
print("\nRMS Energy Statistics:")
print(df_noise['rms'].describe())
print("\nRMS Energy (dB) Statistics:")
print(df_noise['rms_db'].describe())
print("\nSilence Ratio Statistics:")
print(df_noise['silence_ratio'].describe())

# Identify potentially noisy files (high RMS, low silence)
print("\n" + "="*50)
print("Top 10 Noisiest Files (High RMS, Low Silence):")
print("="*50)
df_noisy = df_noise.nlargest(10, 'rms_db')[['filename', 'scenario', 'rms_db', 'silence_ratio']]
print(df_noisy.to_string(index=False))

# Identify potentially quiet files (low RMS, high silence)
print("\n" + "="*50)
print("Top 10 Quietest Files (Low RMS, High Silence):")
print("="*50)
df_quiet = df_noise.nsmallest(10, 'rms_db')[['filename', 'scenario', 'rms_db', 'silence_ratio']]
print(df_quiet.to_string(index=False))