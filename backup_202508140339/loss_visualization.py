#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
損失函數可視化工具
日期: 2025-07-02
功能: 詳細分析和可視化模型訓練過程中的各種損失
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

def plot_detailed_learning_curves(epochs, train_losses, feature_losses, content_losses, save_path=None, experiment_id=None):
    """
    繪製詳細的學習曲線，顯示 total loss、內容一致性損失和 L2 損失
    
    Args:
        epochs (list): 訓練的 epoch 列表
        train_losses (list): 總體訓練損失列表
        feature_losses (list): 特徵 L2 損失列表
        content_losses (list): 內容一致性損失列表
        save_path (str): 保存路徑，若為 None 則自動生成
        experiment_id (str): 實驗 ID，若為 None 則自動生成
        
    Returns:
        str: 保存的圖表路徑
    """
    # 如果沒有提供實驗 ID，生成一個包含日期的實驗 ID
    if experiment_id is None:
        current_date = datetime.now().strftime("%Y%m%d")
        experiment_id = f"EXP{current_date}"
    
    # 如果沒有提供保存路徑，生成一個包含日期和函式名稱的路徑
    if save_path is None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        function_name = "plot_detailed_learning_curves"
        save_dir = os.path.join(os.getcwd(), "results", "loss_curves")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{experiment_id}_{current_date}_{function_name}.png")
    
    plt.figure(figsize=(12, 8))
    
    # 繪製三條損失曲線
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Total Loss')
    plt.plot(epochs, feature_losses, 'g-', linewidth=2, label='L2 Feature Loss')
    
    # 如果有內容一致性損失，則繪製
    if content_losses is not None and len(content_losses) > 0:
        plt.plot(epochs, content_losses, 'r-', linewidth=2, label='Content Consistency Loss')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'訓練損失詳細分析 - {experiment_id}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"詳細學習曲線已保存至 {save_path}")
    return save_path

def plot_loss_comparison(experiment_dirs, labels=None, output_dir=None, output_name=None):
    """
    比較多個實驗的損失曲線
    
    Args:
        experiment_dirs (list): 實驗目錄列表，每個目錄應包含 training_metrics.json 文件
        labels (list): 每個實驗的標籤，如果為 None 則使用目錄名
        output_dir (str): 輸出目錄，如果為 None 則使用當前工作目錄下的 results/comparisons
        output_name (str): 輸出文件名稱，如果為 None 則自動生成
        
    Returns:
        str: 保存的圖表路徑
    """
    if not experiment_dirs:
        print("錯誤：沒有提供實驗目錄")
        return None
    
    if labels is None:
        labels = [os.path.basename(exp_dir) for exp_dir in experiment_dirs]
    
    if len(labels) != len(experiment_dirs):
        print("警告：標籤數量與實驗目錄數量不匹配，將使用目錄名作為標籤")
        labels = [os.path.basename(exp_dir) for exp_dir in experiment_dirs]
    
    # 準備保存路徑
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "results", "comparisons")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if output_name is None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"loss_comparison_{current_date}.png"
    
    output_path = os.path.join(output_dir, output_name)
    
    # 設置子圖
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # 顏色和線型
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    
    # 對每個實驗加載和繪製損失
    for i, (exp_dir, label) in enumerate(zip(experiment_dirs, labels)):
        metrics_file = os.path.join(exp_dir, "training_metrics.json")
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # 提取損失數據
            epochs = metrics.get('epochs', [])
            train_losses = metrics.get('train_losses', [])
            feature_losses = metrics.get('feature_losses', [])
            content_losses = metrics.get('content_losses', [])
            
            color = colors[i % len(colors)]
            linestyle = linestyles[i // len(colors) % len(linestyles)]
            
            # 繪製總損失
            axes[0].plot(epochs, train_losses, color=color, linestyle=linestyle, linewidth=2, label=label)
            axes[0].set_title('Total Loss Comparison', fontsize=14)
            axes[0].set_ylabel('Total Loss', fontsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend(fontsize=11, loc='upper right')
            
            # 繪製特徵 L2 損失
            axes[1].plot(epochs, feature_losses, color=color, linestyle=linestyle, linewidth=2, label=label)
            axes[1].set_title('L2 Feature Loss Comparison', fontsize=14)
            axes[1].set_ylabel('L2 Feature Loss', fontsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            axes[1].legend(fontsize=11, loc='upper right')
            
            # 如果有內容一致性損失，則繪製
            if content_losses:
                axes[2].plot(epochs, content_losses, color=color, linestyle=linestyle, linewidth=2, label=label)
                axes[2].set_title('Content Consistency Loss Comparison', fontsize=14)
                axes[2].set_ylabel('Content Loss', fontsize=12)
                axes[2].grid(True, linestyle='--', alpha=0.7)
                axes[2].legend(fontsize=11, loc='upper right')
            
        except Exception as e:
            print(f"無法讀取實驗 {exp_dir} 的指標文件: {e}")
    
    axes[-1].set_xlabel('Epochs', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"損失比較圖已保存至 {output_path}")
    
    # 更新中央報告
    update_central_report({
        'experiment_id': f"COMP_{datetime.now().strftime('%Y%m%d')}",
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': labels,
        'comparison_type': '損失曲線比較',
        'figure_path': output_path
    })
    
    return output_path

def extract_learning_curves_from_checkpoint(checkpoint_path, save_dir=None):
    """
    從模型檢查點文件中提取學習曲線數據
    
    Args:
        checkpoint_path (str): 檢查點文件路徑
        save_dir (str): 保存提取數據的目錄，如果為 None 則使用檢查點所在目錄
        
    Returns:
        dict: 提取的學習曲線數據
    """
    if not os.path.exists(checkpoint_path):
        print(f"錯誤：檢查點文件 {checkpoint_path} 不存在")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取學習曲線數據
        epochs_record = checkpoint.get('epochs_record', [])
        train_losses_record = checkpoint.get('train_losses_record', [])
        feature_losses_record = checkpoint.get('feature_losses_record', [])
        content_losses_record = checkpoint.get('content_losses_record', [])
        
        # 如果未指定保存目錄，使用檢查點所在目錄
        if save_dir is None:
            save_dir = os.path.dirname(checkpoint_path)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成保存路徑
        experiment_id = os.path.basename(os.path.dirname(checkpoint_path))
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{experiment_id}_{current_date}_learning_curves.json")
        
        # 將數據轉換為 list (tensor無法序列化為 JSON)
        data = {
            'epochs': [int(e) for e in epochs_record],
            'train_losses': [float(l) for l in train_losses_record],
            'feature_losses': [float(l) for l in feature_losses_record],
            'content_losses': [float(l) for l in content_losses_record] if content_losses_record else []
        }
        
        # 保存為 JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"學習曲線數據已提取並保存至 {save_path}")
        
        # 繪製詳細學習曲線
        plot_path = os.path.join(save_dir, f"{experiment_id}_{current_date}_detailed_learning_curves.png")
        plot_detailed_learning_curves(
            data['epochs'], 
            data['train_losses'], 
            data['feature_losses'], 
            data['content_losses'], 
            save_path=plot_path,
            experiment_id=experiment_id
        )
        
        return data
        
    except Exception as e:
        print(f"無法從檢查點文件提取學習曲線: {e}")
        return None

def update_central_report(report_data):
    """
    更新中央報告文件 (REPORT.md)
    
    Args:
        report_data (dict): 報告數據
    """
    report_path = os.path.join(os.getcwd(), "REPORT.md")
    
    # 如果報告文件不存在，創建它
    if not os.path.exists(report_path):
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 實驗報告總匯\n\n")
            f.write("## 損失分析\n\n")
            f.write("| 實驗編號 | 日期 | 類型 | 圖表 |\n")
            f.write("|----------|------|------|------|\n")
    
    # 讀取現有報告
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.readlines()
    
    # 檢查是否有損失分析表格
    loss_analysis_section = False
    table_header_index = -1
    
    for i, line in enumerate(report_content):
        if "## 損失分析" in line:
            loss_analysis_section = True
        elif loss_analysis_section and "| 實驗編號 | 日期 | 類型 | 圖表 |" in line:
            table_header_index = i
            break
    
    # 如果沒有損失分析表格，添加它
    if not loss_analysis_section:
        # 添加在文件末尾
        report_content.append("\n## 損失分析\n\n")
        report_content.append("| 實驗編號 | 日期 | 類型 | 圖表 |\n")
        report_content.append("|----------|------|------|------|\n")
        table_header_index = len(report_content) - 1
    elif table_header_index == -1:
        # 找到損失分析部分但沒有表格，添加表格
        for i, line in enumerate(report_content):
            if "## 損失分析" in line:
                report_content.insert(i + 1, "\n")
                report_content.insert(i + 2, "| 實驗編號 | 日期 | 類型 | 圖表 |\n")
                report_content.insert(i + 3, "|----------|------|------|------|\n")
                table_header_index = i + 3
                break
    
    # 創建新的表格行
    figure_rel_path = os.path.relpath(report_data['figure_path'], os.getcwd())
    
    if 'experiments' in report_data:
        # 比較實驗
        experiments_str = ", ".join(report_data['experiments'])
        new_line = f"| {report_data['experiment_id']} | {report_data['date']} | {report_data['comparison_type']} ({experiments_str}) | [圖表]({figure_rel_path}) |\n"
    else:
        # 單個實驗
        new_line = f"| {report_data['experiment_id']} | {report_data['date']} | 損失分析 | [圖表]({figure_rel_path}) |\n"
    
    # 插入新行
    report_content.insert(table_header_index + 1, new_line)
    
    # 保存更新後的報告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_content)
    
    print(f"已更新中央報告: {report_path}")

def main():
    """主函數，處理命令行參數"""
    parser = argparse.ArgumentParser(description='損失函數可視化工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 提取學習曲線數據的子命令
    extract_parser = subparsers.add_parser('extract', help='從檢查點提取學習曲線數據')
    extract_parser.add_argument('checkpoint', type=str, help='檢查點文件路徑')
    extract_parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    
    # 繪製詳細學習曲線的子命令
    plot_parser = subparsers.add_parser('plot', help='繪製詳細學習曲線')
    plot_parser.add_argument('data_file', type=str, help='學習曲線數據文件路徑 (JSON)')
    plot_parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    plot_parser.add_argument('--experiment_id', type=str, default=None, help='實驗ID')
    
    # 比較多個實驗的子命令
    compare_parser = subparsers.add_parser('compare', help='比較多個實驗的損失曲線')
    compare_parser.add_argument('--experiment_dirs', type=str, nargs='+', required=True, help='實驗目錄列表')
    compare_parser.add_argument('--labels', type=str, nargs='+', default=None, help='實驗標籤列表')
    compare_parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    compare_parser.add_argument('--output_name', type=str, default=None, help='輸出文件名')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_learning_curves_from_checkpoint(args.checkpoint, args.output_dir)
    
    elif args.command == 'plot':
        try:
            with open(args.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            output_dir = args.output_dir if args.output_dir else os.path.dirname(args.data_file)
            os.makedirs(output_dir, exist_ok=True)
            
            experiment_id = args.experiment_id if args.experiment_id else os.path.basename(os.path.dirname(args.data_file))
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{experiment_id}_{current_date}_detailed_learning_curves.png")
            
            plot_detailed_learning_curves(
                data['epochs'], 
                data['train_losses'], 
                data['feature_losses'], 
                data.get('content_losses', []), 
                save_path=output_path,
                experiment_id=experiment_id
            )
        except Exception as e:
            print(f"無法繪製學習曲線: {e}")
    
    elif args.command == 'compare':
        plot_loss_comparison(
            args.experiment_dirs,
            args.labels,
            args.output_dir,
            args.output_name
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
