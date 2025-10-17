"""
WavTokenizer Transformer 系統流程視覺化

生成系統架構和數據流程的可視化圖表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_system_architecture_diagram():
    """創建系統架構圖"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定義顏色
    color_frozen = '#E8F4F8'  # 淺藍色 - 凍結組件
    color_trainable = '#FFE4E1'  # 淺紅色 - 可訓練組件
    color_data = '#F0F8FF'  # 淺灰色 - 數據
    
    # 標題
    ax.text(5, 11.5, 'WavTokenizer Transformer 系統架構', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            fontproperties='SimHei')
    
    # 1. 輸入 - 噪音音頻
    input_box = FancyBboxPatch((1, 10), 8, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=color_data,
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.4, '輸入: 噪音音頻 (Noisy Audio)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    ax.text(5, 10.1, '[batch, 1, time] - 24kHz 採樣率', 
            ha='center', va='center', fontsize=9,
            fontproperties='SimHei')
    
    # 箭頭1
    arrow1 = FancyArrowPatch((5, 10), (5, 9), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # 2. WavTokenizer Encoder (凍結)
    encoder_box = FancyBboxPatch((0.5, 7.5), 9, 1.3,
                                boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor=color_frozen,
                                linewidth=3)
    ax.add_patch(encoder_box)
    ax.text(5, 8.6, 'WavTokenizer Encoder (凍結 ❄️)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    ax.text(5, 8.2, '• 將連續音頻轉換為離散token序列', 
            ha='center', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(5, 7.9, '• 詞彙表大小: 4096 | Token化率: ~75 tokens/秒', 
            ha='center', va='center', fontsize=9,
            fontproperties='SimHei')
    
    # 箭頭2
    arrow2 = FancyArrowPatch((5, 7.5), (5, 6.8), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    ax.text(5.5, 7.1, 'Noisy Tokens', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(5.5, 6.9, '[batch, seq_len]', ha='left', va='center', fontsize=8, style='italic',
            fontproperties='SimHei')
    
    # 3. Transformer 降噪模型 (可訓練)
    transformer_box = FancyBboxPatch((0.5, 4.2), 9, 2.4,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='red', facecolor=color_trainable,
                                    linewidth=3)
    ax.add_patch(transformer_box)
    ax.text(5, 6.3, 'Transformer 降噪模型 (可訓練 🔥)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    
    # Transformer內部結構
    ax.text(5, 5.8, 'Encoder-Decoder 架構', 
            ha='center', va='center', fontsize=10,
            fontproperties='SimHei')
    ax.text(2.5, 5.3, '• Token嵌入層', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(2.5, 5.0, '• 位置編碼 (離散感知)', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(2.5, 4.7, '• 多頭注意力機制', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    
    ax.text(6.5, 5.3, '• 參數量: 89.3M', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(6.5, 5.0, '• 在token空間學習降噪', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(6.5, 4.7, '• 使用Teacher Forcing', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    
    # 箭頭3
    arrow3 = FancyArrowPatch((5, 4.2), (5, 3.5), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    ax.text(5.5, 3.8, 'Denoised Tokens', ha='left', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(5.5, 3.6, '[batch, seq_len]', ha='left', va='center', fontsize=8, style='italic',
            fontproperties='SimHei')
    
    # 4. WavTokenizer Decoder (凍結)
    decoder_box = FancyBboxPatch((0.5, 2), 9, 1.3,
                                boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor=color_frozen,
                                linewidth=3)
    ax.add_patch(decoder_box)
    ax.text(5, 3.1, 'WavTokenizer Decoder (凍結 ❄️)', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    ax.text(5, 2.7, '• 將離散token序列重建為連續音頻', 
            ha='center', va='center', fontsize=9,
            fontproperties='SimHei')
    ax.text(5, 2.4, '• 使用預訓練的VQ-VAE解碼器', 
            ha='center', va='center', fontsize=9,
            fontproperties='SimHei')
    
    # 箭頭4
    arrow4 = FancyArrowPatch((5, 2), (5, 1.2), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # 5. 輸出 - 降噪音頻
    output_box = FancyBboxPatch((1, 0.2), 8, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='#E8F5E9',
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 0.7, '輸出: 降噪音頻 (Denoised Audio) ✓', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            fontproperties='SimHei')
    ax.text(5, 0.4, '[batch, 1, time] - 高質量重建', 
            ha='center', va='center', fontsize=9,
            fontproperties='SimHei')
    
    # 圖例
    frozen_patch = mpatches.Patch(color=color_frozen, label='凍結組件 (預訓練)')
    trainable_patch = mpatches.Patch(color=color_trainable, label='可訓練組件')
    ax.legend(handles=[frozen_patch, trainable_patch], 
             loc='upper right', fontsize=10, prop={'family': 'SimHei'})
    
    plt.tight_layout()
    plt.savefig('system_architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ 系統架構圖已保存: system_architecture_diagram.png")
    plt.close()

def create_training_flow_diagram():
    """創建訓練流程圖"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 標題
    ax.text(8, 11.5, '訓練流程圖 (單個Batch)', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            fontproperties='SimHei')
    
    # 定義步驟
    steps = [
        (2, 10, '1. 數據加載', '加載噪音和乾淨音頻\n[batch, 1, time]'),
        (2, 8.5, '2. 維度標準化', '確保音頻維度正確\nnormalize_audio_dimensions()'),
        (2, 7, '3. Token編碼', 'Encoder: Audio → Tokens\n[batch, seq_len]'),
        (2, 5.5, '4. 準備序列', '添加SOS/EOS特殊token\nTeacher forcing設置'),
        (8, 10, '5. Transformer前向', 'Encoder-Decoder處理\n生成logits'),
        (8, 8.5, '6. 計算損失', '組合token損失\nL2+Consistency+...'),
        (8, 7, '7. 反向傳播', 'loss.backward()\n計算梯度'),
        (8, 5.5, '8. 梯度裁剪', '自適應裁剪\nmax_norm=0.5'),
        (14, 10, '9. 參數更新', 'optimizer.step()\n更新權重'),
        (14, 8.5, '10. Token解碼', 'Decoder: Tokens → Audio\n重建音頻'),
        (14, 7, '11. 記錄指標', '保存損失和梯度範數\n生成樣本'),
        (14, 5.5, '12. 驗證評估', '計算驗證損失\n調整學習率'),
    ]
    
    colors = ['#FFE4E1', '#E8F4F8', '#F0E68C', '#E8F5E9']
    
    for i, (x, y, title, desc) in enumerate(steps):
        color = colors[i % len(colors)]
        box = FancyBboxPatch((x-1, y-0.6), 2.5, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color,
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x+0.25, y+0.3, title, 
                ha='center', va='center', fontsize=10, fontweight='bold',
                fontproperties='SimHei')
        ax.text(x+0.25, y-0.2, desc, 
                ha='center', va='center', fontsize=8,
                fontproperties='SimHei')
    
    # 添加箭頭連接
    arrow_positions = [
        ((2, 9.4), (2, 8.5)),
        ((2, 7.9), (2, 7)),
        ((2, 6.4), (2, 5.5)),
        ((3.5, 5.5), (6.5, 8.5)),
        ((8, 9.4), (8, 8.5)),
        ((8, 7.9), (8, 7)),
        ((8, 6.4), (8, 5.5)),
        ((9.5, 8.5), (12.5, 10)),
        ((14, 9.4), (14, 8.5)),
        ((14, 7.9), (14, 7)),
        ((14, 6.4), (14, 5.5)),
    ]
    
    for start, end in arrow_positions:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)
    
    # 添加循環箭頭
    loop_arrow = mpatches.FancyBboxPatch((1, 4), 13, 0.8,
                                        boxstyle="round,pad=0.1",
                                        edgecolor='red', facecolor='white',
                                        linewidth=2, linestyle='--')
    ax.add_patch(loop_arrow)
    ax.text(7.5, 4.4, '↻ 重複每個batch，累積梯度 (gradient_accumulation_steps=2)', 
            ha='center', va='center', fontsize=10, color='red',
            fontproperties='SimHei')
    
    # 添加Epoch循環
    epoch_box = mpatches.FancyBboxPatch((0.5, 3), 15, 0.8,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='blue', facecolor='white',
                                       linewidth=2, linestyle='--')
    ax.add_patch(epoch_box)
    ax.text(8, 3.4, '↻↻ 重複每個epoch (num_epochs=300)', 
            ha='center', va='center', fontsize=11, color='blue', fontweight='bold',
            fontproperties='SimHei')
    
    # 關鍵修復標註
    fix_box = FancyBboxPatch((0.5, 0.5), 15, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='#F0FFF0',
                            linewidth=2)
    ax.add_patch(fix_box)
    ax.text(8, 2.2, '✅ 關鍵修復項目', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='green',
            fontproperties='SimHei')
    
    fixes = [
        '步驟2: 音頻維度標準化 (修復SConv1d錯誤)',
        '步驟6: 損失權重重新平衡 (coherence: 0.1→0.01)',
        '步驟8: 自適應梯度裁剪 (降低退化率至<30%)',
        '步驟12: 驗證損失修復 (不再返回0)'
    ]
    
    for i, fix in enumerate(fixes):
        ax.text(8, 1.7 - i*0.35, f'• {fix}', 
                ha='center', va='center', fontsize=9,
                fontproperties='SimHei')
    
    plt.tight_layout()
    plt.savefig('training_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ 訓練流程圖已保存: training_flow_diagram.png")
    plt.close()

def create_loss_components_diagram():
    """創建損失函數組件圖"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左圖: 損失組件權重 (修復前vs修復後)
    components = ['L2損失', '一致性損失', 'Manifold\n正則化', '正則化損失', '連貫性損失']
    weights_before = [0.3, 0.4, 0.1, 0.1, 0.1]
    weights_after = [0.4, 0.5, 0.05, 0.04, 0.01]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, weights_before, width, label='修復前', color='#FFB6C1')
    bars2 = ax1.bar(x + width/2, weights_after, width, label='修復後', color='#90EE90')
    
    ax1.set_xlabel('損失組件', fontsize=12, fontproperties='SimHei')
    ax1.set_ylabel('權重', fontsize=12, fontproperties='SimHei')
    ax1.set_title('損失函數權重對比 (修復前 vs 修復後)', fontsize=14, fontweight='bold',
                  fontproperties='SimHei')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontproperties='SimHei')
    ax1.legend(prop={'family': 'SimHei'})
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 右圖: 損失組件說明
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    ax2.text(5, 9.5, '損失函數組件說明', 
            ha='center', va='top', fontsize=16, fontweight='bold',
            fontproperties='SimHei')
    
    descriptions = [
        ('L2損失 (40%)', 
         '計算預測token嵌入與目標token嵌入的歐式距離\n'
         '確保token空間的特徵接近', '#FFE4E1'),
        
        ('一致性損失 (50%)', 
         '主要損失，確保預測準確性和分佈合理性\n'
         '使用CrossEntropy計算token預測錯誤', '#E8F4F8'),
        
        ('Manifold正則化 (5%)', 
         '防止預測偏離輸入manifold太遠\n'
         '保持輸入和輸出的相似性', '#F0E68C'),
        
        ('正則化損失 (4%)', 
         '控制logits的大小和分佈\n'
         '防止過擬合', '#E8F5E9'),
        
        ('連貫性損失 (1%) ⚠️ 修復關鍵', 
         '確保序列的語義連續性\n'
         '已修復: token值歸一化 + 權重大幅降低', '#FFF0F5'),
    ]
    
    y_start = 8.5
    for i, (title, desc, color) in enumerate(descriptions):
        y = y_start - i * 1.6
        
        box = FancyBboxPatch((0.5, y-0.5), 9, 1.3,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color,
                            linewidth=1.5)
        ax2.add_patch(box)
        
        ax2.text(5, y+0.3, title,
                ha='center', va='center', fontsize=11, fontweight='bold',
                fontproperties='SimHei')
        ax2.text(5, y-0.1, desc,
                ha='center', va='center', fontsize=9,
                fontproperties='SimHei')
    
    plt.tight_layout()
    plt.savefig('loss_components_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ 損失函數組件圖已保存: loss_components_diagram.png")
    plt.close()

if __name__ == "__main__":
    print("🎨 開始生成系統可視化圖表...")
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        create_system_architecture_diagram()
        create_training_flow_diagram()
        create_loss_components_diagram()
        
        print("\n✅ 所有可視化圖表生成完成！")
        print("生成的文件:")
        print("  1. system_architecture_diagram.png - 系統架構圖")
        print("  2. training_flow_diagram.png - 訓練流程圖")
        print("  3. loss_components_diagram.png - 損失函數組件圖")
        
    except Exception as e:
        print(f"❌ 生成圖表時出錯: {e}")
        print("提示: 請確保已安裝matplotlib和相關依賴")