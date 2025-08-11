#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
檢查 min_content_samples 參數設置後的特徵形狀

此腳本用於驗證將 min_content_samples 從 3 修改為 5 後的影響，
並檢查增強特徵與目標特徵的形狀。

實驗日期: 2025-08-05
"""

import os
import torch
from datetime import datetime
import argparse

def check_features_shape(features_dir):
    """
    檢查特徵形狀並記錄結果
    
    Args:
        features_dir (str): 特徵文件所在的目錄路徑
        
    Returns:
        dict: 包含特徵形狀信息的字典
    """
    results = {}
    
    try:
        # 加載增強特徵
        enhanced_path = os.path.join(features_dir, 'enhanced_features.pt')
        if os.path.exists(enhanced_path):
            enhanced_features = torch.load(enhanced_path)
            results['enhanced_shape'] = enhanced_features.shape
            results['enhanced_dtype'] = enhanced_features.dtype
            results['enhanced_device'] = enhanced_features.device
            results['enhanced_mean'] = enhanced_features.mean().item()
            results['enhanced_std'] = enhanced_features.std().item()
            print(f"增強特徵形狀: {enhanced_features.shape}")
        else:
            print(f"警告: 找不到增強特徵文件: {enhanced_path}")
    
        # 加載目標特徵
        target_path = os.path.join(features_dir, 'target_features.pt')
        if os.path.exists(target_path):
            target_features = torch.load(target_path)
            results['target_shape'] = target_features.shape
            results['target_dtype'] = target_features.dtype
            results['target_device'] = target_features.device
            results['target_mean'] = target_features.mean().item()
            results['target_std'] = target_features.std().item()
            print(f"目標特徵形狀: {target_features.shape}")
        else:
            print(f"警告: 找不到目標特徵文件: {target_path}")
            
        # 檢查批次大小與最小內容樣本數的比例
        if 'enhanced_shape' in results:
            batch_size = results['enhanced_shape'][0]
            print(f"批次大小: {batch_size}")
            print(f"當前 min_content_samples 設置為: 5")
            print(f"min_content_samples 佔批次大小的比例: {5/batch_size:.2f} ({5}/{batch_size})")
            
    except Exception as e:
        print(f"檢查特徵形狀時出錯: {e}")
        
    return results

def save_report(results, output_dir):
    """
    保存檢查結果報告
    
    Args:
        results (dict): 檢查結果
        output_dir (str): 輸出目錄
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"feature_shape_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 特徵形狀檢查報告 (EXP_PARAM_UPDATE_20250805)\n")
        f.write(f"檢查時間: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 參數設置\n")
        f.write(f"min_content_samples: 5\n\n")
        
        f.write("## 特徵形狀信息\n")
        if 'enhanced_shape' in results:
            f.write(f"增強特徵形狀: {results['enhanced_shape']}\n")
            f.write(f"增強特徵數據類型: {results['enhanced_dtype']}\n")
            f.write(f"增強特徵平均值: {results['enhanced_mean']:.4f}\n")
            f.write(f"增強特徵標準差: {results['enhanced_std']:.4f}\n\n")
        
        if 'target_shape' in results:
            f.write(f"目標特徵形狀: {results['target_shape']}\n")
            f.write(f"目標特徵數據類型: {results['target_dtype']}\n")
            f.write(f"目標特徵平均值: {results['target_mean']:.4f}\n")
            f.write(f"目標特徵標準差: {results['target_std']:.4f}\n\n")
            
        if 'enhanced_shape' in results:
            batch_size = results['enhanced_shape'][0]
            f.write(f"批次大小: {batch_size}\n")
            f.write(f"min_content_samples: 5\n")
            f.write(f"min_content_samples 佔批次大小的比例: {5/batch_size:.2f} ({5}/{batch_size})\n")
            
    print(f"檢查報告已保存到: {report_file}")
    return report_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="檢查特徵形狀")
    parser.add_argument("--features_dir", type=str, 
                      default="/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output2/features/epoch_1200",
                      help="特徵文件目錄路徑")
    parser.add_argument("--output_dir", type=str, 
                      default="/home/sbplab/ruizi/WavTokenize/results/param_updates",
                      help="檢查報告輸出目錄")
    
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 檢查特徵形狀
    results = check_features_shape(args.features_dir)
    
    # 保存檢查報告
    save_report(results, args.output_dir)
