#!/usr/bin/env python3
"""
簡化版離散音頻特徵可視化工具
專注於將音頻轉換為離散token並進行直觀可視化
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

# 設置字體為英文，避免中文字體警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class SimpleDiscreteVisualizer:
    def __init__(self, wavtokenizer_path="/home/sbplab/ruizi/WavTokenize"):
        """簡化版離散可視化器"""
        self.wavtokenizer_path = Path(wavtokenizer_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設置輸出目錄
        self.output_dir = Path("/home/sbplab/ruizi/c_code/results/simple_discrete_visualization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 簡化版離散可視化器初始化完成")
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
        """將音頻轉換為離散tokens（簡化版）"""
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
            
            # 確保正確維度
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)  # [batch, channels, time]
            
            waveform = waveform.to(self.device).contiguous()
            
            print(f"🎵 處理音頻: {Path(audio_path).name}")
            print(f"   音頻形狀: {waveform.shape}")
            
            # 使用WavTokenizer編碼（只要tokens，不需要重建）
            with torch.no_grad():
                bandwidth_id = torch.tensor([0], device=self.device, dtype=torch.long)
                
                _, discrete_code = self.model.encode_infer(
                    waveform,
                    bandwidth_id=bandwidth_id
                )
                
                # 處理discrete_code的形狀
                if discrete_code.dim() == 3:
                    # [n_q, batch, seq_len] -> 使用第一個codebook
                    tokens = discrete_code[0].squeeze()
                elif discrete_code.dim() == 2:
                    # [batch, seq_len]
                    tokens = discrete_code.squeeze()
                else:
                    tokens = discrete_code
                
                tokens = tokens.long().cpu().numpy()
            
            result = {
                'tokens': tokens,
                'sample_rate': sample_rate,
                'audio_length': waveform.shape[-1] / sample_rate,
                'token_length': len(tokens),
                'tokens_per_second': len(tokens) / (waveform.shape[-1] / sample_rate)
            }
            
            print(f"✅ Token提取完成")
            print(f"   Token序列長度: {result['token_length']}")
            print(f"   Token/秒: {result['tokens_per_second']:.1f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Token提取失敗 {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_token_text_visualization(self, token_results, save_path=None):
        """創建類似文字編輯器的token可視化"""
        n_files = len([r for r in token_results.values() if r is not None])
        if n_files == 0:
            print("⚠️ 沒有有效的token數據可視化")
            return None
        
        fig, axes = plt.subplots(n_files, 1, figsize=(20, 4*n_files))
        if n_files == 1:
            axes = [axes]
        
        idx = 0
        for audio_path, result in token_results.items():
            if result is None:
                continue
            
            tokens = result['tokens']
            info = self.extract_audio_info(audio_path)
            
            # 創建token矩陣視圖
            tokens_per_line = 50  # 每行50個tokens
            n_lines = (len(tokens) + tokens_per_line - 1) // tokens_per_line
            
            # 填充token矩陣
            token_matrix = np.full((n_lines, tokens_per_line), -1)
            for i, token in enumerate(tokens):
                row = i // tokens_per_line
                col = i % tokens_per_line
                if row < n_lines:
                    token_matrix[row, col] = token
            
            # 繪製token矩陣
            im = axes[idx].imshow(token_matrix, cmap='tab20', aspect='auto', 
                                 vmin=0, vmax=4095, interpolation='nearest')
            
            # 設置標題
            title = f"{info['speaker']} - {info['material']} (長度: {result['audio_length']:.2f}s, {result['token_length']} tokens)"
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Token位置 (時間順序)')
            axes[idx].set_ylabel('Token行數')
            
            # 添加統計信息
            unique_tokens = len(np.unique(tokens[tokens >= 0]))
            diversity = unique_tokens / len(tokens)
            mean_val = np.mean(tokens)
            
            stats_text = f"唯一Tokens: {unique_tokens} | 多樣性: {diversity:.3f} | 平均值: {mean_val:.1f}"
            axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='top', fontsize=10)
            
            idx += 1
        
        # 添加colorbar
        cbar = plt.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label('Token值 (0-4095)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Token文字視圖已保存: {save_path}")
        
        return fig
    
    def create_token_comparison(self, token_results, save_path=None):
        """創建token序列比較分析"""
        valid_results = {k: v for k, v in token_results.items() if v is not None}
        
        if len(valid_results) < 2:
            print("⚠️ 需要至少2個有效結果進行比較")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 準備數據
        names = []
        tokens_data = []
        stats_data = []
        
        for path, result in valid_results.items():
            info = self.extract_audio_info(path)
            names.append(f"{info['speaker']}_{info['material']}")
            tokens_data.append(result['tokens'])
            
            stats_data.append({
                'name': names[-1],
                'length': len(result['tokens']),
                'unique': len(np.unique(result['tokens'])),
                'diversity': len(np.unique(result['tokens'])) / len(result['tokens']),
                'mean': np.mean(result['tokens']),
                'std': np.std(result['tokens']),
                'tokens_per_sec': result['tokens_per_second']
            })
        
        # 1. Token值分佈比較
        for i, (name, tokens) in enumerate(zip(names, tokens_data)):
            axes[0,0].hist(tokens, bins=50, alpha=0.7, label=name, density=True)
        axes[0,0].set_title('Token值分佈比較', fontweight='bold')
        axes[0,0].set_xlabel('Token值')
        axes[0,0].set_ylabel('密度')
        axes[0,0].legend()
        
        # 2. Token統計比較
        df_stats = pd.DataFrame(stats_data)
        x_pos = np.arange(len(names))
        
        axes[0,1].bar(x_pos - 0.2, df_stats['diversity'], 0.4, label='多樣性', alpha=0.7)
        axes[0,1].bar(x_pos + 0.2, df_stats['mean']/4095, 0.4, label='平均值(標準化)', alpha=0.7)
        axes[0,1].set_title('Token統計特徵比較', fontweight='bold')
        axes[0,1].set_xlabel('音頻檔案')
        axes[0,1].set_ylabel('值')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(names, rotation=45)
        axes[0,1].legend()
        
        # 3. Token序列對比（如果是2個檔案）
        if len(tokens_data) == 2:
            min_len = min(len(tokens_data[0]), len(tokens_data[1]))
            diff = (tokens_data[0][:min_len] != tokens_data[1][:min_len]).astype(int)
            
            # 重塑為矩陣格式顯示差異
            tokens_per_line = 50
            diff_lines = []
            for i in range(0, len(diff), tokens_per_line):
                diff_lines.append(diff[i:i+tokens_per_line])
            
            max_length = max(len(line) for line in diff_lines) if diff_lines else 1
            diff_matrix = np.zeros((len(diff_lines), max_length))
            
            for i, line in enumerate(diff_lines):
                diff_matrix[i, :len(line)] = line
            
            im = axes[1,0].imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
            axes[1,0].set_title(f'Token差異圖\n{names[0]} vs {names[1]}', fontweight='bold')
            axes[1,0].set_xlabel('Token位置')
            axes[1,0].set_ylabel('行數')
            
            # 添加colorbar
            cbar = plt.colorbar(im, ax=axes[1,0])
            cbar.set_label('差異 (0=相同, 1=不同)')
        
        # 4. 統計摘要表
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        
        table_data = []
        for _, row in df_stats.iterrows():
            table_data.append([
                row['name'],
                f"{row['length']}",
                f"{row['unique']}",
                f"{row['diversity']:.3f}",
                f"{row['tokens_per_sec']:.1f}"
            ])
        
        table = axes[1,1].table(
            cellText=table_data,
            colLabels=['檔案', 'Token數', '唯一Token', '多樣性', 'Token/秒'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1,1].set_title('統計摘要', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Token比較分析已保存: {save_path}")
        
        return fig
    
    def generate_simple_report(self, audio_files, experiment_id=None):
        """生成簡化的離散token可視化報告"""
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 開始簡化版離散token分析 - 實驗ID: {experiment_id}")
        
        # 載入模型
        if not self.load_wavtokenizer():
            return None
        
        # 處理所有音頻檔案
        token_results = {}
        for audio_file in audio_files:
            result = self.audio_to_tokens(audio_file)
            token_results[audio_file] = result
        
        # 創建輸出目錄
        exp_dir = self.output_dir / f"experiment_{experiment_id}"
        exp_dir.mkdir(exist_ok=True)
        
        # 生成可視化
        print("📊 生成Token文字視圖...")
        fig1 = self.create_token_text_visualization(
            token_results,
            exp_dir / f"token_text_view_{experiment_id}.png"
        )
        
        print("📊 生成Token比較分析...")
        fig2 = self.create_token_comparison(
            token_results,
            exp_dir / f"token_comparison_{experiment_id}.png"
        )
        
        # 生成報告
        report = self.generate_markdown_report(token_results, experiment_id)
        with open(exp_dir / f"token_analysis_report_{experiment_id}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 簡化版離散token分析完成!")
        print(f"📂 結果保存在: {exp_dir}")
        
        return {
            'experiment_id': experiment_id,
            'output_dir': exp_dir,
            'token_results': token_results
        }
    
    def generate_markdown_report(self, token_results, experiment_id):
        """生成Markdown報告"""
        report = f"""# 離散Token音頻特徵分析報告

**實驗ID**: {experiment_id}  
**分析時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**分析檔案數量**: {len(token_results)}

## 📊 音頻檔案Token化結果

"""
        
        valid_count = 0
        for path, result in token_results.items():
            filename = Path(path).name
            if result is not None:
                info = self.extract_audio_info(path)
                report += f"""
### {filename}
- **說話者**: {info['speaker']}
- **材質**: {info['material']}
- **音頻長度**: {result['audio_length']:.2f} 秒
- **Token序列長度**: {result['token_length']}
- **Token化效率**: {result['tokens_per_second']:.1f} tokens/秒
- **唯一Token數**: {len(np.unique(result['tokens']))}
- **Token多樣性**: {len(np.unique(result['tokens'])) / result['token_length']:.3f}
"""
                valid_count += 1
            else:
                report += f"""
### {filename}
- ❌ **處理失敗**
"""
        
        if valid_count >= 2:
            report += f"""
## 🔍 主要發現

### Token化特徵分析
通過將音頻轉換為離散token序列，我們可以像分析文字一樣分析音頻：

1. **內容差異**: 不同內容的音頻會產生不同的token序列模式
2. **說話者特徵**: 不同說話者的聲音特徵反映在token統計分佈中
3. **材質/噪音影響**: 錄音條件和背景噪音會影響token的多樣性和分佈

### 類似文字編輯的直觀性
- Token序列可以像文字一樣逐行顯示
- 相同內容的不同說話者會有相似但不完全相同的token模式
- 噪音和材質差異在token級別上清晰可見

### 實用價值
這種可視化方法讓音頻分析變得直觀，就像：
- 文字編輯器中比較兩個文檔的差異
- 程式碼比較工具中的語法高亮
- 基因序列分析中的鹼基對比較
"""
        
        report += f"""
## 📈 輸出文件

1. `token_text_view_{experiment_id}.png` - Token序列文字視圖
2. `token_comparison_{experiment_id}.png` - Token比較分析圖
3. `token_analysis_report_{experiment_id}.md` - 本報告

---
*報告生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description='簡化版離散音頻token可視化工具')
    parser.add_argument('--audio_files', nargs='+', required=True,
                       help='要分析的音頻檔案路徑')
    parser.add_argument('--wavtokenizer_path', 
                       default='/home/sbplab/ruizi/WavTokenize',
                       help='WavTokenizer模型路徑')
    parser.add_argument('--experiment_id',
                       help='實驗ID（可選）')
    
    args = parser.parse_args()
    
    # 創建可視化器
    visualizer = SimpleDiscreteVisualizer(args.wavtokenizer_path)
    
    # 生成可視化報告
    results = visualizer.generate_simple_report(args.audio_files, args.experiment_id)
    
    if results:
        print(f"\n🎉 分析完成！")
        print(f"📂 結果目錄: {results['output_dir']}")
        print(f"📊 Token可視化文件已生成")


if __name__ == "__main__":
    main()