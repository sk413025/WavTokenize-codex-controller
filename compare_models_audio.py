#!/usr/bin/env python3
"""
比較新舊模型的音頻品質
- 舊模型: Epoch 300, CE weight = 1.0 (失敗的模型)
- 新模型: Epoch 100, 200, 300, 400, CE weight = 10.0
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path

print("="*70)
print("新舊模型音頻品質比較")
print("="*70)

# 定義要比較的模型
models = {
    "舊模型 (CE=1.0)": {
        "path": "results/transformer_large_tokenloss_large_tokenloss_202510190523/audio_samples/epoch_300",
        "epochs": [300],
        "ce_weight": 1.0,
        "status": "❌ 失敗 (0% token 準確率)"
    },
    "新模型 (CE=10.0)": {
        "path": "results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples",
        "epochs": [100, 200, 300, 400],
        "ce_weight": 10.0,
        "status": "🔄 訓練中 (Epoch 452+)"
    }
}

print("\n" + "="*70)
print("📊 模型配置")
print("="*70)

for model_name, model_info in models.items():
    print(f"\n{model_name}:")
    print(f"   CE Weight: {model_info['ce_weight']}")
    print(f"   Epochs: {model_info['epochs']}")
    print(f"   Status: {model_info['status']}")

print("\n" + "="*70)
print("🎵 音頻品質分析")
print("="*70)

results = {}

for model_name, model_info in models.items():
    print(f"\n{model_name}:")
    results[model_name] = {}
    
    for epoch in model_info['epochs']:
        if model_name == "舊模型 (CE=1.0)":
            audio_dir = Path(model_info['path'])
        else:
            audio_dir = Path(model_info['path']) / f"epoch_{epoch}"
        
        if not audio_dir.exists():
            print(f"   Epoch {epoch}: ❌ 目錄不存在")
            continue
        
        # 分析 enhanced 音頻
        enhanced_files = list(audio_dir.glob("batch_*_enhanced.wav"))
        
        if not enhanced_files:
            print(f"   Epoch {epoch}: ❌ 沒有音頻檔案")
            continue
        
        rms_values = []
        for f in enhanced_files[:5]:  # 只檢查前 5 個
            try:
                audio, _ = torchaudio.load(f)
                rms = torch.sqrt(torch.mean(audio**2)).item()
                rms_values.append(rms)
            except:
                pass
        
        if rms_values:
            avg_rms = np.mean(rms_values)
            std_rms = np.std(rms_values)
            max_rms = np.max(rms_values)
            min_rms = np.min(rms_values)
            
            results[model_name][epoch] = {
                'avg_rms': avg_rms,
                'std_rms': std_rms,
                'max_rms': max_rms,
                'min_rms': min_rms,
                'n_samples': len(rms_values)
            }
            
            # 判斷音頻品質
            if avg_rms < 0.001:
                quality = "❌ 無聲"
            elif avg_rms < 0.01:
                quality = "⚠️  過低"
            elif avg_rms < 0.05:
                quality = "⚠️  偏低"
            elif avg_rms < 0.15:
                quality = "✅ 正常"
            else:
                quality = "✅ 良好"
            
            print(f"   Epoch {epoch:3d}: RMS={avg_rms:.4f} ± {std_rms:.4f}  {quality}")
        else:
            print(f"   Epoch {epoch}: ❌ 無法載入音頻")

print("\n" + "="*70)
print("📈 訓練進度分析")
print("="*70)

# 分析新模型的進度
new_model_results = results.get("新模型 (CE=10.0)", {})
if len(new_model_results) >= 2:
    epochs = sorted(new_model_results.keys())
    print(f"\n新模型 (CE=10.0) 從 Epoch {epochs[0]} 到 Epoch {epochs[-1]}:")
    
    for i in range(len(epochs) - 1):
        ep1, ep2 = epochs[i], epochs[i+1]
        rms1 = new_model_results[ep1]['avg_rms']
        rms2 = new_model_results[ep2]['avg_rms']
        change = ((rms2 - rms1) / rms1) * 100
        
        if abs(change) < 5:
            trend = "→ 穩定"
        elif change > 0:
            trend = "↗ 增強"
        else:
            trend = "↘ 減弱"
        
        print(f"   Epoch {ep1} → {ep2}: {rms1:.4f} → {rms2:.4f} ({change:+.1f}%)  {trend}")

print("\n" + "="*70)
print("🆚 新舊模型對比")
print("="*70)

old_ep300 = results.get("舊模型 (CE=1.0)", {}).get(300)
new_ep100 = results.get("新模型 (CE=10.0)", {}).get(100)
new_ep300 = results.get("新模型 (CE=10.0)", {}).get(300)
new_ep400 = results.get("新模型 (CE=10.0)", {}).get(400)

if old_ep300 and new_ep400:
    print(f"\nEpoch 300 比較:")
    print(f"   舊模型 (CE=1.0): RMS={old_ep300['avg_rms']:.4f}")
    if new_ep300:
        print(f"   新模型 (CE=10.0): RMS={new_ep300['avg_rms']:.4f}")
        diff = ((new_ep300['avg_rms'] - old_ep300['avg_rms']) / old_ep300['avg_rms']) * 100
        print(f"   差異: {diff:+.1f}%")
    
    print(f"\n最新結果 (Epoch 400):")
    print(f"   新模型 (CE=10.0): RMS={new_ep400['avg_rms']:.4f}")
    
    if new_ep100:
        improvement = ((new_ep400['avg_rms'] - new_ep100['avg_rms']) / new_ep100['avg_rms']) * 100
        print(f"   相比 Epoch 100 改善: {improvement:+.1f}%")

print("\n" + "="*70)
print("✅ 結論與建議")
print("="*70)

print("\n1. 音頻品質評估:")
if new_ep400:
    if new_ep400['avg_rms'] > 0.1:
        print("   ✅ Epoch 400 音頻振幅正常，應該有清晰的聲音內容")
    elif new_ep400['avg_rms'] > 0.05:
        print("   ⚠️  Epoch 400 音頻振幅偏低，但應該有可聽的聲音")
    else:
        print("   ❌ Epoch 400 音頻振幅過低，可能仍無法重建人聲")

print("\n2. 下一步行動:")
print("   📁 請手動聽聽這些音頻檔案:")
print(f"      - results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples/epoch_100/batch_0_sample_1_enhanced.wav")
print(f"      - results/transformer_large_tokenloss_large_tokenloss_202510200359/audio_samples/epoch_400/batch_0_sample_1_enhanced.wav")

if old_ep300:
    print(f"      - results/transformer_large_tokenloss_large_tokenloss_202510190523/audio_samples/epoch_300/batch_0_sample_1_enhanced.wav  (舊模型對比)")

print("\n3. 訓練決策:")
print("   🔄 當前訓練: Epoch 452+ 持續進行中")
print("   💡 建議：")
print("      - 先聽 Epoch 400 音頻確認品質")
print("      - 如果品質良好，可以考慮在 Epoch 500-600 停止")
print("      - 如果品質不佳，可能需要調整其他超參數")

print("\n" + "="*70)
