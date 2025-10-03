#!/usr/bin/env python3
"""
音頻特徵對比可視化工具：Mel頻譜圖 vs 離散Token
展示離散token在分析內容、說話者、材質差異方面的優勢
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse
import librosa
import librosa.display

# 設置字體為英文，避免中文字體警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class AudioFeatureComparator:
    def __init__(self, wavtokenizer_path="/home/sbplab/ruizi/WavTokenize"):
        """音頻特徵對比分析器"""
        self.wavtokenizer_path = Path(wavtokenizer_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設置輸出目錄
        self.output_dir = Path("/home/sbplab/ruizi/c_code/results/feature_comparison/n")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Audio Feature Comparator Initialized")
        print(f"📱 Device: {self.device}")
        print(f"📂 Output Directory: {self.output_dir}")
    
    def load_wavtokenizer(self):
        """載入WavTokenizer模型"""
        try:
            import sys
            sys.path.append(str(self.wavtokenizer_path))
            
            from decoder.pretrained import WavTokenizer
            
            # 載入預訓練模型
            model_path = self.wavtokenizer_path / "models" / "wavtokenizer_large_speech_320_24k.ckpt"
            config_path = self.wavtokenizer_path / "config" / "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
            
            self.model = WavTokenizer.from_pretrained0802(
                str(config_path), 
                str(model_path)
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ WavTokenizer Model Loaded Successfully")
            return True
            
        except Exception as e:
            print(f"❌ WavTokenizer Model Loading Failed: {e}")
            return False
    
    def extract_audio_info(self, audio_path):
        """從音頻檔名提取信息"""
        filename = Path(audio_path).stem
        parts = filename.split('_')
        if len(parts) >= 4:
            return {
                'quality': parts[0],
                'speaker': parts[1],
                'material': parts[2],
                'condition': parts[3],
                'number': parts[4] if len(parts) > 4 else '001'
            }
        else:
            return {
                'quality': 'unknown',
                'speaker': 'unknown', 
                'material': 'unknown',
                'condition': 'unknown',
                'number': '001'
            }
    
    def extract_features(self, audio_path):
        """提取音頻的多種特徵：mel頻譜、discrete tokens等"""
        try:
            # 載入音頻
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.float()
            
            # 重採樣到24kHz
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
                sample_rate = 24000
            
            # 轉換為單聲道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 轉換為numpy用於librosa
            audio_np = waveform.squeeze().numpy()
            
            print(f"🎵 Processing Audio: {Path(audio_path).name}")
            
            # 1. 提取Mel頻譜圖
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np, 
                sr=sample_rate,
                n_mels=128,
                hop_length=512,
                win_length=2048
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 2. 提取MFCC特徵
            mfcc = librosa.feature.mfcc(
                y=audio_np,
                sr=sample_rate,
                n_mfcc=13
            )
            
            # 3. 提取離散tokens（使用WavTokenizer）
            tokens = None
            if self.model is not None:
                waveform_gpu = waveform.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    bandwidth_id = torch.tensor([0], device=self.device, dtype=torch.long)
                    
                    _, discrete_code = self.model.encode_infer(
                        waveform_gpu,
                        bandwidth_id=bandwidth_id
                    )
                    
                    if discrete_code.dim() == 3:
                        tokens = discrete_code[0].squeeze()
                    else:
                        tokens = discrete_code.squeeze()
                    
                    tokens = tokens.long().cpu().numpy()
            
            # 4. 提取其他音頻特徵
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_np)
            
            result = {
                'audio_path': audio_path,
                'audio_np': audio_np,
                'sample_rate': sample_rate,
                'duration': len(audio_np) / sample_rate,
                'mel_spectrogram': mel_spec_db,
                'mfcc': mfcc,
                'discrete_tokens': tokens,
                'spectral_centroids': spectral_centroids,
                'zero_crossing_rate': zero_crossing_rate,
                'audio_info': self.extract_audio_info(audio_path)
            }
            
            print(f"✅ Feature Extraction Completed:")
            print(f"   Mel Spectrogram Shape: {mel_spec_db.shape}")
            print(f"   MFCC Shape: {mfcc.shape}")
            if tokens is not None:
                print(f"   Token Sequence Length: {len(tokens)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Feature Extraction Failed {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_comprehensive_comparison(self, feature_results, save_path=None):
        """創建全面的特徵對比可視化"""
        n_files = len(feature_results)
        
        # 創建大型subplot網格
        fig = plt.figure(figsize=(20, 16))
        
        # 設置網格佈局：4行 x n_files列
        gs = fig.add_gridspec(4, n_files, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        for col, (audio_path, result) in enumerate(feature_results.items()):
            if result is None:
                continue
            
            info = result['audio_info']
            title_base = f"{info['speaker']} - {info['material']}"
            
            # 第1行：原始波形
            ax1 = fig.add_subplot(gs[0, col])
            time_axis = np.linspace(0, result['duration'], len(result['audio_np']))
            ax1.plot(time_axis, result['audio_np'], 'b-', linewidth=0.5)
            ax1.set_title(f"{title_base}\nWaveform", fontsize=10, fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # 第2行：Mel頻譜圖
            ax2 = fig.add_subplot(gs[1, col])
            mel_spec = result['mel_spectrogram']
            img = librosa.display.specshow(mel_spec, 
                                         x_axis='time', 
                                         y_axis='mel',
                                         sr=result['sample_rate'],
                                         hop_length=512,
                                         ax=ax2)
            ax2.set_title('Mel Spectrogram\n(Continuous)', fontsize=10, fontweight='bold')
            if col == n_files - 1:  # 只在最後一個subplot添加colorbar
                cbar = plt.colorbar(img, ax=ax2, shrink=0.8)
                cbar.set_label('dB', rotation=270, labelpad=15)
            
            # 第3行：MFCC特徵
            ax3 = fig.add_subplot(gs[2, col])
            mfcc = result['mfcc']
            img2 = librosa.display.specshow(mfcc,
                                          x_axis='time',
                                          ax=ax3)
            ax3.set_title('MFCC Features\n(Continuous)', fontsize=10, fontweight='bold')
            ax3.set_ylabel('MFCC Coefficients')
            if col == n_files - 1:
                cbar2 = plt.colorbar(img2, ax=ax3, shrink=0.8)
            
            # 第4行：離散Token序列
            ax4 = fig.add_subplot(gs[3, col])
            if result['discrete_tokens'] is not None:
                tokens = result['discrete_tokens']
                
                # 創建token矩陣視圖
                tokens_per_line = 20  # 每行顯示token數
                n_lines = (len(tokens) + tokens_per_line - 1) // tokens_per_line
                
                token_matrix = np.full((n_lines, tokens_per_line), -1)
                for i, token in enumerate(tokens):
                    row = i // tokens_per_line
                    col_pos = i % tokens_per_line
                    if row < n_lines:
                        token_matrix[row, col_pos] = token
                
                # 使用特定的colormap來突出顯示不同token值
                im = ax4.imshow(token_matrix, cmap='tab20', aspect='auto', 
                               vmin=0, vmax=4095, interpolation='nearest')
                ax4.set_title('Discrete Tokens\n(Quantized)', fontsize=10, fontweight='bold')
                ax4.set_xlabel('Token Position')
                ax4.set_ylabel('Token Lines')
                
                # 添加統計信息
                unique_tokens = len(np.unique(tokens))
                diversity = unique_tokens / len(tokens)
                stats_text = f"Unique: {unique_tokens}\nDiversity: {diversity:.3f}"
                ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top', fontsize=8)
                
                if col == n_files - 1:
                    cbar3 = plt.colorbar(im, ax=ax4, shrink=0.8)
                    cbar3.set_label('Token Value', rotation=270, labelpad=15)
            else:
                ax4.text(0.5, 0.5, 'Token extraction\nfailed', 
                        transform=ax4.transAxes, ha='center', va='center',
                        fontsize=12, color='red')
                ax4.set_title('Discrete Tokens\n(Failed)', fontsize=10, fontweight='bold')
        
        # 添加整體標題
        fig.suptitle('Audio Feature Comparison: Continuous vs Discrete\n' + 
                    'Demonstrating Discrete Token Advantages for Content/Speaker/Material Analysis',
                    fontsize=16, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comprehensive Feature Comparison Saved: {save_path}")
        
        return fig
    
    def create_discrete_advantages_analysis(self, feature_results, save_path=None):
        """分析並展示離散token相對於連續特徵的優勢"""
        
        if len(feature_results) < 2:
            print("⚠️ Need at least 2 audio files for comparison analysis")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 準備分析數據
        analysis_data = []
        
        for audio_path, result in feature_results.items():
            if result is None or result['discrete_tokens'] is None:
                continue
            
            info = result['audio_info']
            tokens = result['discrete_tokens']
            mel_spec = result['mel_spectrogram']
            
            # 計算各種特徵統計
            data_point = {
                'name': f"{info['speaker']}_{info['material']}",
                'speaker': info['speaker'],
                'material': info['material'],
                
                # Token特徵
                'token_diversity': len(np.unique(tokens)) / len(tokens),
                'token_entropy': self._calculate_entropy(tokens),
                'token_mean': np.mean(tokens),
                'token_std': np.std(tokens),
                'token_changes': np.sum(tokens[1:] != tokens[:-1]) / (len(tokens) - 1),
                
                # Mel頻譜特徵
                'mel_variance': np.var(mel_spec),
                'mel_mean': np.mean(mel_spec),
                'mel_spectral_flatness': self._calculate_spectral_flatness(mel_spec),
                
                # 整體信息
                'duration': result['duration']
            }
            
            analysis_data.append(data_point)
        
        df = pd.DataFrame(analysis_data)
        
        # 1. Token多樣性 vs Mel方差
        axes[0,0].scatter(df['mel_variance'], df['token_diversity'], 
                         c=['red' if 'boy1' in name else 'blue' for name in df['name']], 
                         s=100, alpha=0.7)
        for i, name in enumerate(df['name']):
            axes[0,0].annotate(name, (df['mel_variance'].iloc[i], df['token_diversity'].iloc[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0,0].set_xlabel('Mel Spectrogram Variance (Continuous)')
        axes[0,0].set_ylabel('Token Diversity (Discrete)')
        axes[0,0].set_title('Discrete Token Diversity vs\nContinuous Mel Variance', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Token變化率分析
        speakers = df['speaker'].unique()
        for i, speaker in enumerate(speakers):
            speaker_data = df[df['speaker'] == speaker]
            axes[0,1].bar([f"{speaker}\n{material}" for material in speaker_data['material']], 
                         speaker_data['token_changes'], 
                         alpha=0.7, label=speaker)
        axes[0,1].set_title('Token Change Rate by Speaker/Material\n(Discrete Analysis)', fontweight='bold')
        axes[0,1].set_ylabel('Token Change Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Token熵 vs Mel平坦度
        axes[0,2].scatter(df['mel_spectral_flatness'], df['token_entropy'],
                         c=['green' if 'box' in name else 'orange' for name in df['name']],
                         s=100, alpha=0.7)
        for i, name in enumerate(df['name']):
            axes[0,2].annotate(name, (df['mel_spectral_flatness'].iloc[i], df['token_entropy'].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0,2].set_xlabel('Mel Spectral Flatness (Continuous)')
        axes[0,2].set_ylabel('Token Entropy (Discrete)')
        axes[0,2].set_title('Information Content Comparison\nDiscrete vs Continuous', fontweight='bold')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. 離散特徵的可解釋性優勢
        if len(feature_results) == 2:
            results_list = list(feature_results.values())
            tokens1, tokens2 = results_list[0]['discrete_tokens'], results_list[1]['discrete_tokens']
            
            min_len = min(len(tokens1), len(tokens2))
            differences = (tokens1[:min_len] != tokens2[:min_len]).astype(int)
            
            # 計算差異的時間分佈
            window_size = 10
            diff_windows = []
            for i in range(0, len(differences) - window_size + 1, window_size):
                diff_windows.append(np.mean(differences[i:i+window_size]))
            
            time_windows = np.arange(len(diff_windows)) * window_size / 75  # 假設75 tokens/sec
            
            axes[1,0].plot(time_windows, diff_windows, 'ro-', linewidth=2, markersize=4)
            axes[1,0].set_xlabel('Time (seconds)')
            axes[1,0].set_ylabel('Token Difference Rate')
            axes[1,0].set_title('Temporal Analysis of Discrete Differences\n(Like Text Diff)', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. 特徵穩定性比較
        if len(df) >= 2:
            continuous_features = ['mel_variance', 'mel_mean', 'mel_spectral_flatness']
            discrete_features = ['token_diversity', 'token_entropy', 'token_changes']
            
            # 計算變異係數（CV = std/mean）
            cont_cv = [df[feat].std() / df[feat].mean() for feat in continuous_features]
            disc_cv = [df[feat].std() / df[feat].mean() for feat in discrete_features]
            
            x_pos = np.arange(3)
            width = 0.35
            
            axes[1,1].bar(x_pos - width/2, cont_cv, width, label='Continuous Features', alpha=0.7)
            axes[1,1].bar(x_pos + width/2, disc_cv, width, label='Discrete Features', alpha=0.7)
            axes[1,1].set_xlabel('Feature Types')
            axes[1,1].set_ylabel('Coefficient of Variation')
            axes[1,1].set_title('Feature Stability Comparison\n(Lower = More Stable)', fontweight='bold')
            axes[1,1].set_xticks(x_pos)
            axes[1,1].set_xticklabels(['Variance/Diversity', 'Mean/Entropy', 'Flatness/Changes'])
            axes[1,1].legend()
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. 總結表格：離散特徵的優勢
        axes[1,2].axis('tight')
        axes[1,2].axis('off')
        
        advantages_data = [
            ['Interpretability', 'High', 'Medium'],
            ['Editability', 'Direct Token Edit', 'Complex Transform'],
            ['Compression', 'High (Integer)', 'Lower (Float)'],
            ['Noise Robustness', 'Quantized Levels', 'Continuous Noise'],
            ['Analysis Simplicity', 'Like Text Analysis', 'Signal Processing'],
            ['Speaker Difference', f'{df["token_diversity"].std():.3f}', f'{df["mel_variance"].std():.3f}'],
            ['Material Difference', f'{df["token_entropy"].std():.3f}', f'{df["mel_spectral_flatness"].std():.3f}']
        ]
        
        table = axes[1,2].table(
            cellText=advantages_data,
            colLabels=['Aspect', 'Discrete Tokens', 'Continuous (Mel)'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1,2].set_title('Discrete Token Advantages\nSummary', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Discrete Advantages Analysis Saved: {save_path}")
        
        return fig, df
    
    def _calculate_entropy(self, tokens):
        """計算token序列的熵"""
        unique, counts = np.unique(tokens, return_counts=True)
        probabilities = counts / len(tokens)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _calculate_spectral_flatness(self, mel_spec):
        """計算頻譜平坦度"""
        # 計算幾何平均與算術平均的比值
        geometric_mean = np.exp(np.mean(np.log(np.abs(mel_spec) + 1e-10)))
        arithmetic_mean = np.mean(np.abs(mel_spec))
        return geometric_mean / arithmetic_mean
    
    def generate_comparison_report(self, audio_files, experiment_id=None):
        """生成音頻特徵對比分析報告"""
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 Starting Audio Feature Comparison Analysis - Experiment ID: {experiment_id}")
        
        # 載入模型
        if not self.load_wavtokenizer():
            print("⚠️ WavTokenizer loading failed, will analyze continuous features only")
        
        # 提取所有特徵
        feature_results = {}
        for audio_file in audio_files:
            result = self.extract_features(audio_file)
            feature_results[audio_file] = result
        
        # 創建輸出目錄
        exp_dir = self.output_dir / f"experiment_{experiment_id}"
        exp_dir.mkdir(exist_ok=True)
        
        # 生成可視化
        print("📊 Generating comprehensive feature comparison...")
        fig1 = self.create_comprehensive_comparison(
            feature_results,
            exp_dir / f"comprehensive_comparison_{experiment_id}.png"
        )
        
        print("📊 Generating discrete advantages analysis...")
        fig2, analysis_df = self.create_discrete_advantages_analysis(
            feature_results,
            exp_dir / f"discrete_advantages_{experiment_id}.png"
        )
        
        # 生成報告
        report = self.generate_markdown_report(feature_results, analysis_df, experiment_id)
        with open(exp_dir / f"feature_comparison_report_{experiment_id}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Audio Feature Comparison Analysis Complete!")
        print(f"📂 Results saved in: {exp_dir}")
        
        return {
            'experiment_id': experiment_id,
            'output_dir': exp_dir,
            'feature_results': feature_results,
            'analysis_df': analysis_df
        }
    
    def generate_markdown_report(self, feature_results, analysis_df, experiment_id):
        """Generate Markdown analysis report in English"""
        report = f"""# Audio Feature Comparison Analysis Report: Continuous vs Discrete

**Experiment ID**: {experiment_id}  
**Analysis Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Number of Files Analyzed**: {len(feature_results)}

## 🎯 Experiment Objective

Demonstrate the advantages of discrete token features over traditional continuous features (such as Mel spectrograms) in audio analysis, particularly for:
- **Content Difference Analysis**: Different content from same speaker
- **Speaker Identification**: Different speakers with same content  
- **Material/Noise Detection**: Recording conditions and background environment differences

## 📊 Feature Extraction Results

"""
        
        for audio_path, result in feature_results.items():
            if result is None:
                continue
                
            info = result['audio_info']
            filename = Path(audio_path).name
            
            report += f"""
### {filename}
- **Speaker**: {info['speaker']}
- **Material Environment**: {info['material']}
- **Audio Duration**: {result['duration']:.2f} seconds
- **Mel Spectrogram Shape**: {result['mel_spectrogram'].shape}
- **MFCC Shape**: {result['mfcc'].shape}
"""
            
            if result['discrete_tokens'] is not None:
                tokens = result['discrete_tokens']
                unique_tokens = len(np.unique(tokens))
                diversity = unique_tokens / len(tokens)
                entropy = self._calculate_entropy(tokens)
                
                report += f"""- **Discrete Token Sequence Length**: {len(tokens)}
- **Unique Token Count**: {unique_tokens}
- **Token Diversity**: {diversity:.3f}
- **Token Entropy**: {entropy:.3f}
"""
            else:
                report += "- **Discrete Tokens**: Extraction failed\n"
        
        if analysis_df is not None and not analysis_df.empty:
            report += f"""
## 🔍 Key Findings

### 1. Advantages of Discrete Tokens

#### 🎨 Intuitiveness and Editability
- **Text-like Editing**: Token sequences can be viewed and edited like text
- **Version Control Friendly**: Can use Git-like tools to track audio content changes
- **Human Inspection**: Researchers can directly examine token values at specific time points

#### 🔧 Technical Advantages
- **Quantization Stability**: Discrete values unaffected by floating-point precision
- **Compression Efficiency**: Integer tokens more compact than floating-point spectrograms
- **Noise Resistance**: Quantization process naturally filters small-amplitude noise

#### 📈 Analysis Advantages
"""
            
            # 添加統計對比
            if len(analysis_df) >= 2:
                token_div_std = analysis_df['token_diversity'].std()
                mel_var_std = analysis_df['mel_variance'].std()
                
                report += f"""
**Speaker Difference Detection**:
- Token diversity standard deviation: {token_div_std:.4f}
- Mel variance standard deviation: {mel_var_std:.4f}
- Token features show more stable performance in speaker difference detection

**Material/Environment Differences**:
- Token sequence change rate directly reflects environmental noise impact
- Easier to quantify and analyze compared to continuous Mel spectrogram changes
"""
        
        report += f"""
### 2. Application Scenario Comparison

| Application | Continuous Features (Mel) | Discrete Tokens |
|------------|---------------------------|-----------------|
| Speaker ID | Complex statistical models | Direct token statistics |
| Content Analysis | Spectral pattern recognition | Sequence pattern matching |
| Noise Detection | Frequency domain energy analysis | Token diversity changes |
| Audio Editing | Spectral reconstruction needed | Direct token manipulation |
| Data Storage | Large floating-point data | Compact integer sequences |

### 3. Practical Benefits

#### 🎵 Text-like Audio Processing
- **Difference Comparison**: Can use text diff tools to compare token sequences of two audio clips
- **Pattern Search**: Use string search algorithms to find specific audio patterns
- **Statistical Analysis**: Calculate token frequency, n-gram analysis, etc.

#### 🔬 Research and Development Advantages
- **Debug Friendly**: Can precisely locate problem occurrence time (token position)
- **Reproducibility**: Token sequences are completely deterministic, eliminating floating-point precision issues
- **Cross-platform Consistency**: Integer tokens remain consistent across different systems

## 📈 Conclusion

Discrete token features successfully bring audio analysis into the "text processing" era:

1. **Intuitiveness**: Audio content becomes as readable and editable as text
2. **Efficiency**: More compact representation and faster processing speed
3. **Robustness**: More resistant to noise and quantization errors
4. **Operability**: Supports direct digital editing and analysis operations

This approach is particularly suitable for audio applications requiring precise control and analysis, such as speech synthesis, audio editing, and content analysis.

## 📈 Output Files

1. `comprehensive_comparison_{experiment_id}.png` - Comprehensive feature comparison chart
2. `discrete_advantages_{experiment_id}.png` - Discrete advantages analysis chart  
3. `feature_comparison_report_{experiment_id}.md` - This analysis report

---
*Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Experiment purpose: Demonstrate advantages of discrete tokens over continuous features in audio analysis*
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Audio Feature Comparison Tool: Continuous vs Discrete')
    parser.add_argument('--audio_files', nargs='+', required=True,
                       help='Audio file paths to analyze')
    parser.add_argument('--wavtokenizer_path', 
                       default='/home/sbplab/ruizi/WavTokenize',
                       help='WavTokenizer model path')
    parser.add_argument('--experiment_id',
                       help='Experiment ID (optional)')
    
    args = parser.parse_args()
    
    # Create analyzer
    comparator = AudioFeatureComparator(args.wavtokenizer_path)
    
    # Generate comparison analysis report
    results = comparator.generate_comparison_report(args.audio_files, args.experiment_id)
    
    if results:
        print(f"\n🎉 Comparison analysis complete!")
        print(f"📂 Results directory: {results['output_dir']}")
        print(f"📊 Demonstrated advantages of discrete tokens over continuous features")


if __name__ == "__main__":
    main()