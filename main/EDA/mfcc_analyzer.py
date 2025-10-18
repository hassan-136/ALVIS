import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class MFCCAnalyzer:
    def __init__(self, audio_path, n_mfcc=13, sr=22050):
        """
        Initialize MFCC Analyzer for Humdial project
        
        Parameters:
        - audio_path: path to audio file
        - n_mfcc: number of MFCC coefficients (default: 13)
        - sr: sampling rate (default: 22050 Hz)
        """
        self.audio_path = audio_path
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.y = None
        self.mfcc = None
        self.spectrogram = None
        
    def load_audio(self):
        """Load audio file"""
        print(f"Loading audio file: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        print(f"Audio loaded. Duration: {len(self.y)/self.sr:.2f} seconds")
        
    def extract_mfcc(self):
        """Extract MFCC features"""
        print("Extracting MFCC features...")
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=self.n_mfcc)
        print(f"MFCC shape: {self.mfcc.shape}")
        
    def compute_spectrogram(self):
        """Compute spectrogram"""
        print("Computing spectrogram...")
        self.spectrogram = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        self.spectrogram_db = librosa.power_to_db(self.spectrogram, ref=np.max)
        
    def plot_mfcc_features(self, output_path='mfcc_feature_plot.png'):
        """Generate MFCC feature plot"""
        print("Generating MFCC feature plot...")
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(self.mfcc, x_axis='time', sr=self.sr, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features - Humdial Project')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"MFCC plot saved: {output_path}")
        plt.close()
        
    def plot_spectrogram(self, output_path='spectrogram.png'):
        """Generate spectrogram image"""
        print("Generating spectrogram image...")
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(self.spectrogram_db, x_axis='time', y_axis='mel', 
                                sr=self.sr, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram - Humdial Project')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved: {output_path}")
        plt.close()
        
    def generate_stats_csv(self, output_path='mfcc_stats.csv'):
        """Generate statistics CSV file"""
        print("Generating MFCC statistics...")
        
        stats = {
            'mfcc_coefficient': [],
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'median': []
        }
        
        for i in range(self.mfcc.shape[0]):
            stats['mfcc_coefficient'].append(f'MFCC_{i+1}')
            stats['mean'].append(np.mean(self.mfcc[i]))
            stats['std'].append(np.std(self.mfcc[i]))
            stats['min'].append(np.min(self.mfcc[i]))
            stats['max'].append(np.max(self.mfcc[i]))
            stats['median'].append(np.median(self.mfcc[i]))
        
        df = pd.DataFrame(stats)
        df.to_csv(output_path, index=False)
        print(f"Statistics saved: {output_path}")
        print(f"\nStatistics preview:\n{df.head()}")
        
    def run_full_analysis(self, output_dir='output'):
        """Run complete MFCC analysis pipeline"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Run analysis
        self.load_audio()
        self.extract_mfcc()
        self.compute_spectrogram()
        
        # Generate outputs
        self.plot_mfcc_features(f'{output_dir}/mfcc_feature_plot.png')
        self.plot_spectrogram(f'{output_dir}/spectrogram.png')
        self.generate_stats_csv(f'{output_dir}/mfcc_stats.csv')
        
        print("\n" + "="*50)
        print("MFCC Analysis Complete!")
        print(f"All outputs saved in '{output_dir}/' directory")
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "your_audio_file.wav"  # Change this to your audio file
    
    # Initialize analyzer
    analyzer = MFCCAnalyzer(audio_file, n_mfcc=13, sr=22050)
    
    # Run complete analysis
    analyzer.run_full_analysis(output_dir='humdial_output')
    
    # Or run individual steps:
    # analyzer.load_audio()
    # analyzer.extract_mfcc()
    # analyzer.compute_spectrogram()
    # analyzer.plot_mfcc_features('mfcc_plot.png')
    # analyzer.plot_spectrogram('spectrogram.png')
    # analyzer.generate_stats_csv('stats.csv')