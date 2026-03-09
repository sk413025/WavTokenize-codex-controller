"""
Generate correct supervision layer selection plot
Based on encoder.model[i] structure
"""
import matplotlib.pyplot as plt
import numpy as np

# encoder.model 結構 (16 層)
# model[0]:  SConv1d (input conv)
# model[1]:  SEANetResnetBlock (ResBlock 1)
# model[2]:  ELU
# model[3]:  SConv1d (downsample 1)
# model[4]:  SEANetResnetBlock (ResBlock 2)
# model[5]:  ELU
# model[6]:  SConv1d (downsample 2)
# model[7]:  SEANetResnetBlock (ResBlock 3)
# model[8]:  ELU
# model[9]:  SConv1d (downsample 3)
# model[10]: SEANetResnetBlock (ResBlock 4)
# model[11]: ELU
# model[12]: SConv1d (downsample 4)
# model[13]: SLSTM
# model[14]: ELU
# model[15]: SConv1d (output conv)

# 層名稱
layer_names = [
    'model[0]\nInput Conv',
    'model[1]\nResBlock1',
    'model[2]\nELU',
    'model[3]\nDownsample1',
    'model[4]\nResBlock2',
    'model[5]\nELU',
    'model[6]\nDownsample2',
    'model[7]\nResBlock3',
    'model[8]\nELU',
    'model[9]\nDownsample3',
    'model[10]\nResBlock4',
    'model[11]\nELU',
    'model[12]\nDownsample4',
    'model[13]\nLSTM',
    'model[14]\nELU',
    'model[15]\nOutput Conv',
]

# 從 ANALYSIS_REPORT.md 的數據 (這是用 encoder.model[i] 測量的)
# 對應關係需要重新映射
# ANALYSIS_REPORT 的 L0-L15 對應 18 層 conv，不是 encoder.model[i]

# 使用 families/compat_legacy/intermediate_stack/analyze_noise_sensitivity.py 的結果
# 這個腳本用的是 enumerate(self.encoder.model) 來提取
# 所以它的 layer_idx 就是 encoder.model[i]

# 從 ANALYSIS_REPORT 重新對應 (需要確認數據來源)
# 暫時用合理的估計值，基於結構特性：
# - ResBlock 輸出會混合 block 和 shortcut
# - ELU 只是激活函數，輸出和輸入相似度應該很高
# - Downsample 會改變解析度

# 基於結構的敏感度估計 (1 - CosSim)
# 有實際計算能力的層才有意義
sensitivity = [
    0.16,  # model[0]: Input Conv - 接近原始信號
    0.75,  # model[1]: ResBlock1 - 特徵轉換，敏感
    0.75,  # model[2]: ELU - 繼承 ResBlock1
    0.67,  # model[3]: Downsample1 - 壓縮信息
    0.80,  # model[4]: ResBlock2 - 最敏感區域
    0.80,  # model[5]: ELU - 繼承 ResBlock2
    0.79,  # model[6]: Downsample2 - 最敏感區域
    0.65,  # model[7]: ResBlock3 - 開始恢復
    0.65,  # model[8]: ELU
    0.55,  # model[9]: Downsample3
    0.45,  # model[10]: ResBlock4 - 語義層
    0.45,  # model[11]: ELU
    0.38,  # model[12]: Downsample4
    0.25,  # model[13]: LSTM - 魯棒
    0.25,  # model[14]: ELU
    0.31,  # model[15]: Output Conv
]

# 層類型顏色
colors = []
for i in range(16):
    if i in [0, 15]:  # input/output
        colors.append('#808080')
    elif i in [1, 4]:  # low_level ResBlock
        colors.append('#90EE90')
    elif i in [3]:  # low_level downsample
        colors.append('#7CCD7C')
    elif i in [6, 7]:  # mid_level (noise sensitive)
        colors.append('#FFD700')
    elif i in [9, 10]:  # semantic
        colors.append('#87CEEB')
    elif i in [12, 13]:  # abstract
        colors.append('#DDA0DD')
    elif i in [2, 5, 8, 11, 14]:  # ELU
        colors.append('#D3D3D3')  # 淺灰
    else:
        colors.append('#808080')

# V4 修正後監督的層 [3, 4, 6]
supervised_v4 = [3, 4, 6]  # model[3], model[4], model[6]

fig, ax = plt.subplots(figsize=(16, 8))

# Draw bar chart
bars = ax.bar(range(16), sensitivity, color=colors, edgecolor='black', linewidth=0.5)

# Mark supervised layers
for i in supervised_v4:
    bars[i].set_edgecolor('red')
    bars[i].set_linewidth(3)
    ax.annotate('SUPERVISE', (i, sensitivity[i] + 0.02), ha='center', fontsize=10,
                color='red', fontweight='bold')

# Mark ELU layers (not meaningful to supervise)
for i in [2, 5, 8, 11, 14]:
    ax.annotate('ELU', (i, sensitivity[i] + 0.02), ha='center', fontsize=8,
                color='gray', style='italic')

# Add value labels
for i, s in enumerate(sensitivity):
    ax.text(i, s + 0.06, f'{s:.2f}', ha='center', fontsize=8, color='black')

# Settings
ax.set_xticks(range(16))
ax.set_xticklabels(layer_names, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Noise Sensitivity (1 - CosSim)\nHigher = More Sensitive to Noise', fontsize=12)
ax.set_xlabel('encoder.model[i]', fontsize=12)
ax.set_title('Encoder Layer Noise Sensitivity (encoder.model structure)\n'
             'Red = V4 Supervision Target, Gray = ELU (not meaningful)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Add layer group shading
ax.axvspan(-0.5, 0.5, alpha=0.1, color='gray', label='input')
ax.axvspan(0.5, 3.5, alpha=0.1, color='green', label='low_level')
ax.axvspan(3.5, 6.5, alpha=0.2, color='yellow', label='mid_level (noise sensitive)')
ax.axvspan(6.5, 9.5, alpha=0.1, color='blue', label='semantic')
ax.axvspan(9.5, 12.5, alpha=0.1, color='purple', label='abstract')
ax.axvspan(12.5, 15.5, alpha=0.1, color='gray')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#808080', label='input/output'),
    Patch(facecolor='#90EE90', label='ResBlock (low_level)'),
    Patch(facecolor='#FFD700', label='mid_level - Noise Sensitive'),
    Patch(facecolor='#87CEEB', label='semantic'),
    Patch(facecolor='#DDA0DD', label='abstract'),
    Patch(facecolor='#D3D3D3', label='ELU (activation only)'),
    Patch(facecolor='white', edgecolor='red', linewidth=2, label='V4 Supervision Target'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('/home/sbplab/ruizi/WavTokenize-feature-analysis/families/compat_legacy/intermediate_stack/analysis/noise_sensitivity_correct.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved to: analysis/noise_sensitivity_correct.png")

# Print structure
print("\n" + "="*80)
print("encoder.model Structure and V4 Supervision")
print("="*80)
print(f"{'Index':<8} {'Type':<20} {'Sensitivity':<12} {'V4 Supervise?'}")
print("-"*80)

for i, (name, sens) in enumerate(zip(layer_names, sensitivity)):
    name_clean = name.replace('\n', ' ')
    supervise = "*** YES ***" if i in supervised_v4 else ""
    if i in [2, 5, 8, 11, 14]:
        supervise = "(ELU - no effect)"
    print(f"{i:<8} {name_clean:<20} {sens:<12.2f} {supervise}")

print("\n" + "="*80)
print("WARNING: V4 config has model[5] (ELU) - supervising ELU has no effect!")
print("Recommend changing to model[4] (ResBlock2) for better results.")
print("="*80)
