#!/usr/bin/env python3
"""
離散音頻特徵可視化工具
將音頻轉換為離散token並進行可視化分析，類似文字編輯的直觀性
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DiscreteAudioVisualizer:
    def __init__(self, wavtokenizer_path="/home/sbplab/ruizi/WavTokenize"):
        """
        離散音頻特徵可視化器初始化
        
        Args:
            wavtokenizer_path: WavTokenizer模型路徑
        """
        self.wavtokenizer_path = Path(wavtokenizer_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設置輸出目錄
        self.output_dir = Path("/home/sbplab/ruizi/c_code/results/discrete_visualization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 離散音頻可視化器初始化完成")
        print(f"📱 設備: {self.device}")
        print(f"📂 輸出目錄: {self.output_dir}")
    
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
            
            print(f"✅ WavTokenizer模型載入成功")
            return True
            
        except Exception as e:
            print(f"❌ WavTokenizer模型載入失敗: {e}")
            return False
    
    def extract_audio_info(self, audio_path):
        """從音頻檔名提取信息"""
        filename = Path(audio_path).stem
        
        # 解析檔名格式: nor_boy1_box_LDV_001
        parts = filename.split('_')
        if len(parts) >= 4:
            info = {
                'quality': parts[0],        # nor
                'speaker': parts[1],        # boy1, boy2, girl1, etc.
                'material': parts[2],       # box, clean, etc.
                'condition': parts[3],      # LDV
                'number': parts[4] if len(parts) > 4 else '001'
            }
        else:
            info = {
                'quality': 'unknown',
                'speaker': 'unknown', 
                'material': 'unknown',
                'condition': 'unknown',
                'number': '001'
            }
        
        return info
    
    def audio_to_tokens(self, audio_path):
        """將音頻轉換為離散tokens"""
        try:
            # 載入音頻
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 確保數據類型為float32
            waveform = waveform.float()
            
            # 重採樣到24kHz (如果需要)
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
                sample_rate = 24000
            
            # 轉換為單聲道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 確保張量維度正確 [1, channels, time]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            
            # 移動到GPU並確保連續內存
            waveform = waveform.to(self.device).contiguous()
            
            print(f"🎵 音頻形狀: {waveform.shape}, 類型: {waveform.dtype}")
            
            # 使用WavTokenizer編碼
            with torch.no_grad():
                # 創建bandwidth_id張量
                bandwidth_id = torch.tensor([0], device=self.device, dtype=torch.long)
                
                # 編碼
                features, discrete_code = self.model.encode_infer(
                    waveform,
                    bandwidth_id=bandwidth_id
                )
                
                # 處理discrete_code的數據類型和形狀
                if discrete_code.dim() == 3:
                    # 形狀通常是 [n_q, batch_size, seq_len]，使用第一個codebook
                    discrete_tokens = discrete_code[0].squeeze().long()
                else:
                    discrete_tokens = discrete_code.squeeze().long()
                
                # 確保discrete_code為long類型用於decode
                discrete_code_for_decode = discrete_code.long()
                
                # 使用codes_to_features進行轉換（避免直接decode的類型問題）
                features_from_codes = self.model.codes_to_features(discrete_code_for_decode)
                
                # 解碼驗證
                reconstructed = self.model.decode(
                    features_from_codes, 
                    bandwidth_id=bandwidth_id
                )
            
            # 轉換為numpy
            discrete_tokens_np = discrete_tokens.cpu().numpy()
            continuous_features = features.squeeze().cpu().numpy()
            reconstructed_audio = reconstructed.squeeze().cpu().numpy()
            original_audio = waveform.squeeze().cpu().numpy()
            
            # 計算重建品質
            mse = np.mean((original_audio - reconstructed_audio)**2)
            snr = 10 * np.log10(np.var(original_audio) / mse) if mse > 0 else float('inf')
            
            result = {
                'discrete_tokens': discrete_tokens_np,
                'continuous_features': continuous_features,
                'reconstructed_audio': reconstructed_audio,
                'original_audio': original_audio,
                'sample_rate': sample_rate,
                'snr_db': snr,
                'token_sequence_length': discrete_tokens_np.shape[-1] if discrete_tokens_np.ndim > 0 else 1,
                'feature_dim': continuous_features.shape[0] if continuous_features.ndim > 1 else 1
            }
            
            print(f"✅ 音頻轉換完成: {Path(audio_path).name}")
            print(f"   Token序列長度: {result['token_sequence_length']}")
            print(f"   重建SNR: {snr:.2f} dB")
            
            return result
            
        except Exception as e:
            print(f"❌ 音頻轉換失敗 {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_token_text_view(self, audio_results, save_path=None):
        """創建類似文字編輯器的token視圖"""
        fig, axes = plt.subplots(len(audio_results), 1, figsize=(20, 4*len(audio_results)))
        if len(audio_results) == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 為不同token值分配顏色
        
        for idx, (audio_path, result) in enumerate(audio_results.items()):
            if result is None:
                continue
                
            tokens = result['discrete_tokens']
            info = self.extract_audio_info(audio_path)
            
            # 將tokens reshape為類似文字的行列格式
            tokens_per_line = 50  # 每行顯示50個tokens
            token_lines = []
            for i in range(0, len(tokens), tokens_per_line):
                token_lines.append(tokens[i:i+tokens_per_line])
            
            # 創建token矩陣可視化
            max_length = max(len(line) for line in token_lines)
            token_matrix = np.full((len(token_lines), max_length), -1)
            
            for i, line in enumerate(token_lines):
                token_matrix[i, :len(line)] = line
            
            # 繪製token矩陣
            im = axes[idx].imshow(token_matrix, cmap='tab20', aspect='auto', vmin=0, vmax=4095)
            
            # 設置標題和標籤
            title = f"{info['speaker']} - {info['material']} (SNR: {result['snr_db']:.1f}dB)"
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Token位置 (時間順序)')
            axes[idx].set_ylabel('Token行數')
            
            # 添加token統計信息
            unique_tokens = len(np.unique(tokens))
            token_diversity = unique_tokens / len(tokens)
            
            # 在圖上添加統計信息
            stats_text = f"總Tokens: {len(tokens)} | 唯一Tokens: {unique_tokens} | 多樣性: {token_diversity:.3f}"
            axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Token文字視圖已保存: {save_path}")
        
        return fig
    
    def create_token_difference_map(self, audio_results, save_path=None):
        """創建token差異熱力圖，突出顯示內容/說話者/材質的差異"""
        valid_results = {k: v for k, v in audio_results.items() if v is not None}
        
        if len(valid_results) < 2:
            print("⚠️ 至少需要2個有效的音頻檔案才能進行差異分析")
            return None
        
        # 準備數據
        audio_paths = list(valid_results.keys())
        token_sequences = []
        labels = []
        
        for path, result in valid_results.items():
            tokens = result['discrete_tokens']
            info = self.extract_audio_info(path)
            
            token_sequences.append(tokens)
            labels.append(f"{info['speaker']}_{info['material']}")
        
        # 對齊token序列長度
        min_length = min(len(seq) for seq in token_sequences)
        aligned_sequences = [seq[:min_length] for seq in token_sequences]
        
        # 計算token差異矩陣
        n_files = len(aligned_sequences)
        diff_matrix = np.zeros((n_files, n_files))
        
        for i in range(n_files):
            for j in range(n_files):
                if i != j:
                    # 計算token序列的差異度
                    diff_ratio = np.sum(aligned_sequences[i] != aligned_sequences[j]) / min_length
                    diff_matrix[i, j] = diff_ratio
        
        # 創建熱力圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 差異熱力圖
        sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title('Token序列差異矩陣', fontsize=14, fontweight='bold')
        ax1.set_xlabel('音頻檔案')
        ax1.set_ylabel('音頻檔案')
        
        # Token位置差異圖
        if len(aligned_sequences) == 2:
            position_diff = (aligned_sequences[0] != aligned_sequences[1]).astype(int)
            
            # 重塑為矩陣格式
            tokens_per_line = 50
            diff_lines = []
            for i in range(0, len(position_diff), tokens_per_line):
                diff_lines.append(position_diff[i:i+tokens_per_line])
            
            max_length = max(len(line) for line in diff_lines)
            diff_matrix_vis = np.full((len(diff_lines), max_length), 0)
            
            for i, line in enumerate(diff_lines):
                diff_matrix_vis[i, :len(line)] = line
            
            im = ax2.imshow(diff_matrix_vis, cmap='RdYlBu_r', aspect='auto')
            ax2.set_title(f'Token位置差異圖\n{labels[0]} vs {labels[1]}', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Token位置')
            ax2.set_ylabel('Token行數')
            
            # 添加顏色條
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('差異 (0=相同, 1=不同)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Token差異圖已保存: {save_path}")
        
        return fig
    
    def create_token_pattern_analysis(self, audio_results, save_path=None):
        """分析token模式，識別內容、說話者、材質的特徵模式"""
        
        # 準備數據用於分析
        analysis_data = []
        
        for path, result in audio_results.items():
            if result is None:
                continue
                
            tokens = result['discrete_tokens']
            info = self.extract_audio_info(path)
            
            # 計算token統計特徵
            token_stats = {
                'file': Path(path).name,
                'speaker': info['speaker'],
                'material': info['material'],
                'total_tokens': len(tokens),
                'unique_tokens': len(np.unique(tokens)),
                'token_diversity': len(np.unique(tokens)) / len(tokens),
                'mean_token_value': np.mean(tokens),
                'std_token_value': np.std(tokens),
                'max_token_value': np.max(tokens),
                'min_token_value': np.min(tokens),
                'snr_db': result['snr_db']
            }
            
            # 計算token轉換頻率 (相鄰token變化的頻率)
            token_changes = np.sum(tokens[1:] != tokens[:-1])
            token_stats['change_frequency'] = token_changes / (len(tokens) - 1)
            
            # 計算最常見的token值
            unique, counts = np.unique(tokens, return_counts=True)
            most_common_idx = np.argmax(counts)
            token_stats['most_common_token'] = unique[most_common_idx]
            token_stats['most_common_ratio'] = counts[most_common_idx] / len(tokens)
            
            analysis_data.append(token_stats)
        
        # 轉換為DataFrame便於分析
        df = pd.DataFrame(analysis_data)
        
        # 創建可視化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Token多樣性 vs 說話者
        speakers = df['speaker'].unique()
        materials = df['material'].unique()
        
        for i, speaker in enumerate(speakers):
            speaker_data = df[df['speaker'] == speaker]
            axes[0, 0].scatter(speaker_data['material'], speaker_data['token_diversity'], 
                             label=speaker, s=100, alpha=0.7)
        
        axes[0, 0].set_title('Token多樣性 vs 說話者/材質', fontweight='bold')
        axes[0, 0].set_xlabel('材質')
        axes[0, 0].set_ylabel('Token多樣性')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Token變化頻率 vs 材質
        for material in materials:
            material_data = df[df['material'] == material]
            axes[0, 1].boxplot([material_data['change_frequency']], 
                              positions=[list(materials).index(material)],
                              widths=0.6, patch_artist=True)
        
        axes[0, 1].set_title('Token變化頻率 vs 材質', fontweight='bold')
        axes[0, 1].set_xlabel('材質')
        axes[0, 1].set_ylabel('Token變化頻率')
        axes[0, 1].set_xticks(range(len(materials)))
        axes[0, 1].set_xticklabels(materials, rotation=45)
        
        # 3. Token平均值 vs 說話者
        speaker_colors = plt.cm.Set1(np.linspace(0, 1, len(speakers)))
        for i, speaker in enumerate(speakers):
            speaker_data = df[df['speaker'] == speaker]
            axes[0, 2].scatter(speaker_data['mean_token_value'], speaker_data['std_token_value'],
                             label=speaker, c=[speaker_colors[i]], s=100, alpha=0.7)
        
        axes[0, 2].set_title('Token統計特徵 (平均值 vs 標準差)', fontweight='bold')
        axes[0, 2].set_xlabel('Token平均值')
        axes[0, 2].set_ylabel('Token標準差')
        axes[0, 2].legend()
        
        # 4. 最常見Token vs 說話者
        axes[1, 0].scatter(df['speaker'], df['most_common_token'], 
                          c=df['most_common_ratio'], s=100, alpha=0.7, cmap='viridis')
        axes[1, 0].set_title('最常見Token vs 說話者', fontweight='bold')
        axes[1, 0].set_xlabel('說話者')
        axes[1, 0].set_ylabel('最常見Token值')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. SNR vs Token多樣性
        material_colors = {material: plt.cm.Set2(i/len(materials)) 
                          for i, material in enumerate(materials)}
        for material in materials:
            material_data = df[df['material'] == material]
            axes[1, 1].scatter(material_data['snr_db'], material_data['token_diversity'],
                             label=material, c=[material_colors[material]], s=100, alpha=0.7)
        
        axes[1, 1].set_title('重建品質(SNR) vs Token多樣性', fontweight='bold')
        axes[1, 1].set_xlabel('SNR (dB)')
        axes[1, 1].set_ylabel('Token多樣性')
        axes[1, 1].legend()
        
        # 6. 統計摘要表格
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        summary_stats = df.groupby(['speaker', 'material']).agg({
            'token_diversity': 'mean',
            'change_frequency': 'mean',
            'snr_db': 'mean'
        }).round(3)
        
        table_data = []
        for (speaker, material), row in summary_stats.iterrows():
            table_data.append([speaker, material, 
                             f"{row['token_diversity']:.3f}",
                             f"{row['change_frequency']:.3f}",
                             f"{row['snr_db']:.1f}"])
        
        table = axes[1, 2].table(cellText=table_data,
                               colLabels=['說話者', '材質', 'Token多樣性', '變化頻率', 'SNR(dB)'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('統計摘要', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Token模式分析圖已保存: {save_path}")
        
        return fig, df
    
    def create_interactive_token_explorer(self, audio_results, save_path=None):
        """創建互動式token探索器 (使用Plotly)"""
        
        # 準備數據
        all_tokens = []
        all_labels = []
        all_colors = []
        all_info = []
        
        color_map = {
            'boy1': 'blue', 'boy2': 'lightblue',
            'girl1': 'red', 'girl2': 'pink',
            'box': 'brown', 'clean': 'green'
        }
        
        for path, result in audio_results.items():
            if result is None:
                continue
                
            tokens = result['discrete_tokens']
            info = self.extract_audio_info(path)
            
            # 為每個token添加標籤和信息
            for i, token in enumerate(tokens):
                all_tokens.append(token)
                all_labels.append(f"{info['speaker']}_{info['material']}")
                all_colors.append(color_map.get(info['speaker'], 'gray'))
                all_info.append({
                    'position': i,
                    'speaker': info['speaker'],
                    'material': info['material'],
                    'file': Path(path).name,
                    'token_value': token
                })
        
        # 使用PCA進行降維可視化
        if len(all_tokens) > 100:
            # 將tokens重塑為特徵向量
            token_vectors = []
            window_size = 10  # 使用滑動窗口創建特徵向量
            
            for path, result in audio_results.items():
                if result is None:
                    continue
                    
                tokens = result['discrete_tokens']
                info = self.extract_audio_info(path)
                
                for i in range(0, len(tokens) - window_size + 1, window_size//2):
                    window = tokens[i:i+window_size]
                    if len(window) == window_size:
                        token_vectors.append(window)
                        all_info.append({
                            'speaker': info['speaker'],
                            'material': info['material'],
                            'file': Path(path).name,
                            'position': i
                        })
            
            # PCA降維
            pca = PCA(n_components=2)
            token_vectors_2d = pca.fit_transform(token_vectors)
            
            # 創建互動式散點圖
            fig = go.Figure()
            
            for speaker in set(info['speaker'] for info in all_info[-len(token_vectors):]):
                speaker_mask = [info['speaker'] == speaker for info in all_info[-len(token_vectors):]]
                speaker_points = token_vectors_2d[speaker_mask]
                speaker_info = [info for info, mask in zip(all_info[-len(token_vectors):], speaker_mask) if mask]
                
                fig.add_trace(go.Scatter(
                    x=speaker_points[:, 0],
                    y=speaker_points[:, 1],
                    mode='markers',
                    name=speaker,
                    text=[f"檔案: {info['file']}<br>材質: {info['material']}<br>位置: {info['position']}" 
                          for info in speaker_info],
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Token特徵空間可視化 (PCA)',
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                hovermode='closest'
            )
            
            if save_path:
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                print(f"📊 互動式Token探索器已保存: {html_path}")
        
        return fig if 'fig' in locals() else None
    
    def generate_report(self, audio_files, experiment_id=None):
        """生成完整的離散特徵可視化報告"""
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 開始離散特徵可視化分析 - 實驗ID: {experiment_id}")
        
        # 載入模型
        if not self.load_wavtokenizer():
            return None
        
        # 處理所有音頻檔案
        audio_results = {}
        for audio_file in audio_files:
            result = self.audio_to_tokens(audio_file)
            audio_results[audio_file] = result
        
        # 創建輸出目錄
        exp_dir = self.output_dir / f"experiment_{experiment_id}"
        exp_dir.mkdir(exist_ok=True)
        
        # 生成各種可視化
        visualizations = {}
        
        # 1. Token文字視圖
        print("📊 生成Token文字視圖...")
        fig1 = self.create_token_text_view(
            audio_results, 
            exp_dir / f"token_text_view_{experiment_id}.png"
        )
        visualizations['token_text_view'] = fig1
        
        # 2. Token差異圖
        print("📊 生成Token差異分析...")
        fig2 = self.create_token_difference_map(
            audio_results,
            exp_dir / f"token_difference_map_{experiment_id}.png"
        )
        visualizations['token_difference_map'] = fig2
        
        # 3. Token模式分析
        print("📊 生成Token模式分析...")
        fig3, analysis_df = self.create_token_pattern_analysis(
            audio_results,
            exp_dir / f"token_pattern_analysis_{experiment_id}.png"
        )
        visualizations['token_pattern_analysis'] = fig3
        
        # 4. 互動式探索器
        print("📊 生成互動式Token探索器...")
        fig4 = self.create_interactive_token_explorer(
            audio_results,
            exp_dir / f"interactive_token_explorer_{experiment_id}.png"
        )
        visualizations['interactive_explorer'] = fig4
        
        # 保存分析數據
        analysis_df.to_csv(exp_dir / f"token_analysis_data_{experiment_id}.csv", index=False)
        
        # 生成總結報告
        report = self.generate_summary_report(audio_results, analysis_df, experiment_id)
        with open(exp_dir / f"analysis_report_{experiment_id}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 離散特徵可視化分析完成!")
        print(f"📂 結果保存在: {exp_dir}")
        
        return {
            'experiment_id': experiment_id,
            'output_dir': exp_dir,
            'visualizations': visualizations,
            'analysis_data': analysis_df,
            'audio_results': audio_results
        }
    
    def generate_summary_report(self, audio_results, analysis_df, experiment_id):
        """生成總結報告"""
        report = f"""# 離散音頻特徵可視化分析報告

**實驗ID**: {experiment_id}  
**分析時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**分析檔案數量**: {len(audio_results)}

## 📊 分析摘要

### 音頻檔案信息
"""
        
        for path, result in audio_results.items():
            if result is None:
                continue
            info = self.extract_audio_info(path)
            report += f"""
- **{Path(path).name}**
  - 說話者: {info['speaker']}
  - 材質: {info['material']}
  - Token序列長度: {result['token_sequence_length']}
  - 重建SNR: {result['snr_db']:.2f} dB
"""
        
        # 添加統計分析
        if not analysis_df.empty:
            report += f"""
### 🔍 主要發現

#### Token多樣性分析
- 平均Token多樣性: {analysis_df['token_diversity'].mean():.3f}
- Token多樣性範圍: {analysis_df['token_diversity'].min():.3f} - {analysis_df['token_diversity'].max():.3f}

#### 說話者差異
"""
            for speaker in analysis_df['speaker'].unique():
                speaker_data = analysis_df[analysis_df['speaker'] == speaker]
                report += f"- **{speaker}**: 平均多樣性 {speaker_data['token_diversity'].mean():.3f}, 平均變化頻率 {speaker_data['change_frequency'].mean():.3f}\n"
            
            report += f"""
#### 材質影響分析
"""
            for material in analysis_df['material'].unique():
                material_data = analysis_df[analysis_df['material'] == material]
                report += f"- **{material}**: 平均多樣性 {material_data['token_diversity'].mean():.3f}, 平均SNR {material_data['snr_db'].mean():.1f}dB\n"
        
        report += f"""
### 📈 可視化文件

1. `token_text_view_{experiment_id}.png` - Token序列文字視圖
2. `token_difference_map_{experiment_id}.png` - Token差異熱力圖  
3. `token_pattern_analysis_{experiment_id}.png` - Token模式統計分析
4. `interactive_token_explorer_{experiment_id}.html` - 互動式Token探索器

### 🎯 結論

離散Token特徵成功將音頻轉換為類似文字的序列，能夠：
- **內容差異**: 通過Token序列模式識別
- **說話者差異**: 通過Token統計特徵區分
- **材質/噪音差異**: 通過Token變化頻率和多樣性體現

這種可視化方法提供了直觀的音頻分析視角，類似於文字編輯器中的文本分析。

---
*報告生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description='離散音頻特徵可視化工具')
    parser.add_argument('--audio_files', nargs='+', required=True,
                       help='要分析的音頻檔案路徑')
    parser.add_argument('--wavtokenizer_path', 
                       default='/home/sbplab/ruizi/WavTokenizer',
                       help='WavTokenizer模型路徑')
    parser.add_argument('--experiment_id',
                       help='實驗ID（可選）')
    
    args = parser.parse_args()
    
    # 創建可視化器
    visualizer = DiscreteAudioVisualizer(args.wavtokenizer_path)
    
    # 生成可視化報告
    results = visualizer.generate_report(args.audio_files, args.experiment_id)
    
    if results:
        print(f"\n🎉 分析完成！")
        print(f"📂 結果目錄: {results['output_dir']}")
        print(f"📊 可視化文件已生成")


if __name__ == "__main__":
    main()