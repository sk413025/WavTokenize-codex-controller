"""
Token Accuracy 反推分析工具

用途: 從整體 accuracy 反推各 token 群組的準確率

實驗日期: 2025-11-05
相關報告: PLATEAU_MECHANISM_ANALYSIS.md, Section "數學推導"
"""

import numpy as np


def analyze_token_accuracy(
    overall_acc,
    token_453_ratio,
    dataset_name="Dataset"
):
    """
    從整體準確率反推 Token 453 和其他 tokens 的可能準確率
    
    公式: Overall_Acc = (1 - Token453_Ratio) × Other_Acc + Token453_Ratio × Token453_Acc
    
    Args:
        overall_acc: 整體準確率 (e.g., 0.547 for 54.7%)
        token_453_ratio: Token 453 在數據集中的佔比 (e.g., 0.1357 for 13.57%)
        dataset_name: 數據集名稱
    """
    print(f"\n{'='*70}")
    print(f"{dataset_name} Accuracy 分析")
    print(f"{'='*70}\n")
    
    print(f"已知:")
    print(f"  整體準確率: {overall_acc*100:.2f}%")
    print(f"  Token 453 佔比: {token_453_ratio*100:.2f}%")
    print(f"  其他 tokens 佔比: {(1-token_453_ratio)*100:.2f}%")
    
    # 情境 1: Token 453 完全失敗
    print(f"\n情境 1: 如果 Token 453 完全失敗 (0% 準確率)")
    other_acc_scenario1 = overall_acc / (1 - token_453_ratio)
    print(f"  → 其他 tokens 準確率 = {other_acc_scenario1*100:.2f}%")
    
    # 情境 2: Token 453 有不同的準確率
    print(f"\n情境 2: 不同的 Token 453 準確率假設")
    for token_453_acc in [0.0, 0.2, 0.4, 0.6]:
        other_acc = (overall_acc - token_453_ratio * token_453_acc) / (1 - token_453_ratio)
        print(f"  Token 453 準確率 = {token_453_acc*100:.0f}% → 其他 tokens = {other_acc*100:.2f}%")
    
    # 情境 3: Token 453 對錯誤的貢獻
    error_rate = 1 - overall_acc
    token_453_max_error_contribution = token_453_ratio / error_rate * 100
    
    print(f"\n情境 3: Token 453 對整體錯誤的最大貢獻")
    print(f"  整體錯誤率: {error_rate*100:.2f}%")
    print(f"  如果 Token 453 完全錯誤，佔總錯誤: {token_453_max_error_contribution:.1f}%")
    
    return {
        'overall_acc': overall_acc,
        'token_453_ratio': token_453_ratio,
        'error_rate': error_rate,
        'max_error_contribution': token_453_max_error_contribution
    }


def compare_train_val_accuracy(
    train_acc,
    val_acc,
    train_token_453_ratio,
    val_token_453_ratio
):
    """
    比較 Train 和 Val 的準確率差異，分析是否來自 Token 453
    
    Args:
        train_acc: 訓練集準確率 (e.g., 0.547)
        val_acc: 驗證集準確率 (e.g., 0.3675)
        train_token_453_ratio: Token 453 在 Train 的佔比 (e.g., 0.1357)
        val_token_453_ratio: Token 453 在 Val 的佔比 (e.g., 0.1865)
    """
    print(f"\n{'='*70}")
    print(f"Train vs Val Accuracy Gap 分析")
    print(f"{'='*70}\n")
    
    gap = train_acc - val_acc
    print(f"Train-Val Gap: {gap*100:.2f}% ({train_acc*100:.2f}% - {val_acc*100:.2f}%)")
    
    # 分析 Train
    train_stats = analyze_token_accuracy(train_acc, train_token_453_ratio, "Train Set")
    
    # 分析 Val
    val_stats = analyze_token_accuracy(val_acc, val_token_453_ratio, "Val Set")
    
    # 假設: 其他 tokens 的準確率在 Train/Val 相同
    print(f"\n{'='*70}")
    print(f"假設驗證: 其他 tokens 在 Train/Val 準確率相同")
    print(f"{'='*70}\n")
    
    # 嘗試不同的 other_acc 值
    for other_acc in [0.60, 0.55, 0.50, 0.45]:
        token_453_acc_train = (train_acc - (1 - train_token_453_ratio) * other_acc) / train_token_453_ratio
        token_453_acc_val = (val_acc - (1 - val_token_453_ratio) * other_acc) / val_token_453_ratio
        
        print(f"如果 Other_Acc = {other_acc*100:.0f}%:")
        print(f"  Train Token 453 Acc = {token_453_acc_train*100:+.2f}% {'✓' if token_453_acc_train >= 0 else '❌ 負值！'}")
        print(f"  Val Token 453 Acc   = {token_453_acc_val*100:+.2f}% {'✓' if token_453_acc_val >= 0 else '❌ 負值！'}")
        print()
    
    # 結論
    print(f"{'='*70}")
    print(f"關鍵發現:")
    print(f"{'='*70}\n")
    
    # 如果 Token 453 完全失敗，其他 tokens 的準確率
    train_other_acc_max = train_acc / (1 - train_token_453_ratio)
    val_other_acc_max = val_acc / (1 - val_token_453_ratio)
    other_acc_gap = train_other_acc_max - val_other_acc_max
    
    print(f"1. 如果 Token 453 完全失敗 (0%):")
    print(f"   Train 其他 tokens 準確率: {train_other_acc_max*100:.2f}%")
    print(f"   Val 其他 tokens 準確率:   {val_other_acc_max*100:.2f}%")
    print(f"   差異: {other_acc_gap*100:.2f}%")
    print(f"\n   → 說明: 即使 Token 453 表現相同，其他 tokens 在 Val 也下降了 {other_acc_gap*100:.1f}%")
    print(f"   → 結論: 問題不只是 Token 453，整體分布都有 mismatch\n")
    
    print(f"2. Token 453 對錯誤的貢獻:")
    print(f"   Train: 最多佔 {train_stats['max_error_contribution']:.1f}% 的錯誤")
    print(f"   Val:   最多佔 {val_stats['max_error_contribution']:.1f}% 的錯誤")
    print(f"\n   → 結論: Token 453 是瓶頸，但無法完全解釋 {gap*100:.0f}% 的 gap\n")


def main():
    """主函式：執行完整的 accuracy 反推分析"""
    
    # 實際數據 (來自訓練 log Epoch 32)
    train_acc = 0.5470  # 54.70%
    val_acc = 0.3675    # 36.75%
    train_token_453_ratio = 0.1357  # 13.57%
    val_token_453_ratio = 0.1865    # 18.65%
    
    # 執行分析
    compare_train_val_accuracy(
        train_acc,
        val_acc,
        train_token_453_ratio,
        val_token_453_ratio
    )
    
    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
