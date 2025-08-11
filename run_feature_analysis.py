#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WavTokenizer 特徵分析執行腳本
- 用於執行特徵分析工具
- 支援命令行參數

實驗編號: EXP09_RUN
日期: 2025-08-12
作者: GitHub Copilot
"""

import os
import sys
import argparse
from datetime import datetime
from feature_analysis_tool import FeatureAnalyzer, analyze_tsne_outputs


def run_task(task_name: str = "檢查特徵形狀") -> None:
    """
    執行預定義的任務
    
    Args:
        task_name: 要執行的任務名稱
    """
    if task_name == "檢查特徵形狀":
        import torch
        enhanced_features = torch.load('/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output2/features/epoch_1200/enhanced_features.pt')
        target_features = torch.load('/home/sbplab/ruizi/WavTokenize/results/tsne_outputs/output2/features/epoch_1200/target_features.pt')
        
        print(f'Enhanced features shape: {enhanced_features.shape}')
        print(f'Target features shape: {target_features.shape}')
        
        # 自動執行分析
        output_dir = f'results/feature_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        analyzer = FeatureAnalyzer(output_dir=output_dir)
        
        # 分析特徵分佈
        analyzer.analyze_feature_distribution(enhanced_features, name='enhanced')
        analyzer.analyze_feature_distribution(target_features, name='target')
        
        # 執行t-SNE降維
        analyzer.perform_tsne(enhanced_features, name='enhanced')
        analyzer.perform_tsne(target_features, name='target')
        
        # 執行聚類分析
        analyzer.perform_clustering(enhanced_features, name='enhanced')
        analyzer.perform_clustering(target_features, name='target')
        
        # 比較兩組特徵
        analyzer.compare_features(
            enhanced_features, 
            target_features,
            name_a='enhanced',
            name_b='target',
            comparison_name='enhanced_vs_target'
        )
        
        print(f"特徵分析完成，結果已保存至: {analyzer.output_dir}")


def main() -> None:
    """
    主函數，解析命令行參數並執行相應的操作
    """
    parser = argparse.ArgumentParser(description='WavTokenizer 特徵分析執行腳本')
    parser.add_argument('--output-dir', type=str,
                        help='輸出目錄路徑')
    parser.add_argument('--enhanced-features', type=str,
                        help='增強特徵檔案路徑')
    parser.add_argument('--target-features', type=str,
                        help='目標特徵檔案路徑')
    parser.add_argument('--tsne-output-dir', type=str,
                        help='t-SNE輸出目錄路徑')
    parser.add_argument('--experiment-id', type=str,
                        help='實驗ID，如果未提供則自動生成')
    parser.add_argument('--task', type=str, default='檢查特徵形狀',
                        help='執行預定義的任務')
    
    args = parser.parse_args()
    
    if args.task and not (args.enhanced_features or args.target_features or args.tsne_output_dir):
        # 執行預定義的任務
        run_task(args.task)
    elif args.tsne_output_dir:
        # 分析t-SNE輸出目錄
        analyze_tsne_outputs(
            output_dir=args.tsne_output_dir,
            experiment_id=args.experiment_id,
            enhanced_features_path=args.enhanced_features,
            target_features_path=args.target_features
        )
    elif args.enhanced_features and args.target_features:
        output_dir = args.output_dir or f'results/feature_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 創建特徵分析器
        analyzer = FeatureAnalyzer(
            output_dir=output_dir,
            experiment_id=args.experiment_id
        )
        
        # 載入特徵
        enhanced_features = analyzer.load_features(args.enhanced_features)
        target_features = analyzer.load_features(args.target_features)
        
        # 分析特徵分佈
        analyzer.analyze_feature_distribution(enhanced_features, name='enhanced')
        analyzer.analyze_feature_distribution(target_features, name='target')
        
        # 執行t-SNE降維
        analyzer.perform_tsne(enhanced_features, name='enhanced')
        analyzer.perform_tsne(target_features, name='target')
        
        # 比較兩組特徵
        analyzer.compare_features(
            enhanced_features, 
            target_features,
            name_a='enhanced',
            name_b='target',
            comparison_name='enhanced_vs_target'
        )
        
        print(f"特徵分析完成，結果已保存至: {analyzer.output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
