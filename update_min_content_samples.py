#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
更新min_content_samples參數的腳本

此腳本用於執行參數更新並記錄到REPORT.md
"""

import os
import sys

# 設置環境變量，讓程序能找到模塊
sys.path.append("/home/sbplab/ruizi/WavTokenize")

from exp_update_report import generate_report_entry

if __name__ == "__main__":
    # 定義實驗配置
    exp_config = {
        'parameter_name': 'min_content_samples',
        'old_value': 3,
        'new_value': 5,
    }
    
    # 定義實驗結果指標
    metrics = {}
    
    # 定義實驗描述
    description = "增加批次中相同內容ID的最小樣本數，以提高內容一致性損失的計算效果，並強化模型對內容不變特徵的學習"
    
    # 生成報告條目
    generate_report_entry(
        exp_id="PARAM_UPDATE",
        exp_name="min_content_samples參數更新",
        exp_config=exp_config,
        results_dir="",
        metrics=metrics,
        description=description,
    )
