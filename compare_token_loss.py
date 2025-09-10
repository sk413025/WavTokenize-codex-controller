#!/usr/bin/env python3
"""
測試 Token Loss 系統的效果
比較原始交叉熵 vs Token Loss（ttt2.py 風格）的訓練效果
"""

import os
import subprocess
import argparse
import logging

def run_training_comparison(epochs=10, max_samples=100):
    """
    運行訓練比較實驗
    
    實驗編號：EXP-20250908-001
    實驗目的：比較交叉熵損失與 Token Loss 系統（基於 ttt2.py 邏輯）的效果
    """
    
    base_cmd = [
        "python", "discrete_token_denoising.py",
        "--num_epochs", str(epochs),
        "--max_samples", str(max_samples),
        "--batch_size", "4",  # 減小批次以便快速測試
        "--save_every", "5"
    ]
    
    experiments = [
        {
            "name": "CrossEntropy",
            "output_dir": "results/token_loss_comparison/crossentropy",
            "extra_args": [],
            "description": "使用原始交叉熵損失"
        },
        {
            "name": "TokenLoss_Balanced",
            "output_dir": "results/token_loss_comparison/token_loss_balanced",
            "extra_args": [
                "--use_token_loss",
                "--l2_weight", "0.3",
                "--consistency_weight", "0.4", 
                "--manifold_weight", "0.1",
                "--normalization_weight", "0.1",
                "--coherence_weight", "0.1"
            ],
            "description": "使用 Token Loss 系統，平衡權重"
        },
        {
            "name": "TokenLoss_ConsistencyFocus",
            "output_dir": "results/token_loss_comparison/token_loss_consistency",
            "extra_args": [
                "--use_token_loss",
                "--l2_weight", "0.2",
                "--consistency_weight", "0.6",  # 更專注於一致性
                "--manifold_weight", "0.1", 
                "--normalization_weight", "0.05",
                "--coherence_weight", "0.05"
            ],
            "description": "使用 Token Loss 系統，專注內容一致性"
        },
        {
            "name": "TokenLoss_L2Focus",
            "output_dir": "results/token_loss_comparison/token_loss_l2",
            "extra_args": [
                "--use_token_loss",
                "--l2_weight", "0.5",  # 更專注於 L2 距離
                "--consistency_weight", "0.3",
                "--manifold_weight", "0.1",
                "--normalization_weight", "0.05", 
                "--coherence_weight", "0.05"
            ],
            "description": "使用 Token Loss 系統，專注 L2 距離"
        }
    ]
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('results/token_loss_comparison.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Token Loss 系統比較實驗開始")
    logging.info(f"實驗編號：EXP-20250908-001")
    logging.info(f"實驗背景：測試將 ttt2.py 的 loss 邏輯應用到 token 空間的效果")
    logging.info(f"實驗動機：比較不同 loss 設計對 token 序列降噪的影響")
    logging.info(f"實驗目的：驗證 L2、內容一致性、正則化、manifold、連貫性 loss 在 token 空間的有效性")
    logging.info(f"訓練輪數：{epochs}")
    logging.info(f"最大樣本數：{max_samples}")
    logging.info("=" * 60)
    
    results = {}
    
    for exp in experiments:
        logging.info(f"\n開始實驗：{exp['name']}")
        logging.info(f"描述：{exp['description']}")
        logging.info(f"輸出目錄：{exp['output_dir']}")
        
        # 創建輸出目錄
        os.makedirs(exp['output_dir'], exist_ok=True)
        
        # 構建命令
        cmd = base_cmd + ["--output_dir", exp['output_dir']] + exp['extra_args']
        
        logging.info(f"執行命令：{' '.join(cmd)}")
        
        try:
            # 執行訓練
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logging.info(f"實驗 {exp['name']} 成功完成")
                results[exp['name']] = {
                    'status': 'success',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                logging.error(f"實驗 {exp['name']} 失敗")
                logging.error(f"錯誤輸出：{result.stderr}")
                results[exp['name']] = {
                    'status': 'failed',
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logging.error(f"實驗 {exp['name']} 超時")
            results[exp['name']] = {
                'status': 'timeout',
                'stdout': '',
                'stderr': 'Training timeout after 1 hour'
            }
        except Exception as e:
            logging.error(f"實驗 {exp['name']} 執行錯誤：{e}")
            results[exp['name']] = {
                'status': 'error',
                'stdout': '',
                'stderr': str(e)
            }
    
    # 生成實驗報告
    generate_comparison_report(results, epochs, max_samples)
    
    return results

def generate_comparison_report(results, epochs, max_samples):
    """生成比較實驗報告"""
    
    report_path = "results/token_loss_comparison_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Token Loss 系統比較實驗報告\n\n")
        f.write(f"**實驗編號：** EXP-20250908-001\n")
        f.write(f"**實驗日期：** 2025-09-08\n")
        f.write(f"**實驗參數：** epochs={epochs}, max_samples={max_samples}\n\n")
        
        f.write("## 實驗背景\n")
        f.write("將 ttt2.py 中的 loss 運算邏輯（L2 距離、內容一致性、正則化、manifold 約束、連貫性）應用到離散 token 空間，")
        f.write("測試這種方法對 token 序列降噪任務的效果。\n\n")
        
        f.write("## 實驗動機\n") 
        f.write("原始的交叉熵損失只關注 token 預測的準確性，而 ttt2.py 的 loss 設計考慮了更多語義和結構約束。")
        f.write("我們希望驗證這些約束在 token 空間是否同樣有效。\n\n")
        
        f.write("## 實驗目的\n")
        f.write("1. 比較不同 loss 設計對降噪效果的影響\n")
        f.write("2. 驗證 L2 距離在 token 嵌入空間的作用\n")
        f.write("3. 測試 manifold 正則化對 token 序列的約束效果\n")
        f.write("4. 評估連貫性損失對序列質量的提升\n\n")
        
        f.write("## 實驗結果\n\n")
        
        for name, result in results.items():
            f.write(f"### {name}\n")
            f.write(f"**狀態：** {result['status']}\n")
            
            if result['status'] == 'success':
                f.write("**訓練成功完成**\n")
                # 可以添加更多分析，如從 stdout 解析 loss 值
            elif result['status'] == 'failed':
                f.write(f"**訓練失敗**\n")
                f.write(f"錯誤信息：\n```\n{result['stderr'][:500]}\n```\n")
            else:
                f.write(f"**訓練{result['status']}**\n")
            
            f.write("\n")
        
        f.write("## 實驗解讀\n")
        f.write("### 預期結果\n")
        f.write("1. Token Loss 系統應該在驗證準確率上優於純交叉熵\n")
        f.write("2. 專注 L2 距離的配置可能在重建質量上表現更好\n")
        f.write("3. 專注一致性的配置可能在 token 預測準確率上表現更好\n")
        f.write("4. 平衡配置應該在整體性能上達到較好的綜合效果\n\n")
        
        f.write("### 實際執行結果\n")
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        f.write(f"- 成功完成的實驗：{success_count}/{len(results)}\n")
        f.write(f"- 失敗的實驗：{len(results) - success_count}/{len(results)}\n\n")
        
        f.write("## 實驗反思\n")
        if success_count > 0:
            f.write("1. Token Loss 系統成功應用了 ttt2.py 的 loss 邏輯\n")
            f.write("2. 不同權重配置對訓練穩定性的影響需要進一步分析\n")
            f.write("3. 嵌入層的選擇對 L2 距離計算的影響是關鍵因素\n")
        else:
            f.write("1. 需要檢查 Token Loss 系統的實現是否正確\n") 
            f.write("2. 可能需要調整權重配置以提高訓練穩定性\n")
            f.write("3. 嵌入層的獲取方式可能需要優化\n")
        
        f.write("\n## 下次實驗建議\n")
        f.write("1. 如果效果好，可以嘗試更大的數據集和更長的訓練\n")
        f.write("2. 可以嘗試動態調整權重的方案\n")
        f.write("3. 可以添加更多評估指標，如 BLEU、音頻質量評估等\n")
        f.write("4. 考慮將最佳配置應用到實際音頻降噪任務中\n\n")
        
        f.write("## 重現實驗步驟\n")
        f.write("```bash\n")
        f.write("# 1. 確保環境設置正確\n")
        f.write("cd /home/sbplab/ruizi/c_code\n\n")
        f.write("# 2. 運行比較實驗\n")
        f.write("python compare_token_loss.py --epochs 10 --max_samples 100\n\n")
        f.write("# 3. 查看結果\n")
        f.write("ls results/token_loss_comparison/\n")
        f.write("cat results/token_loss_comparison_report.md\n")
        f.write("```\n")
    
    logging.info(f"實驗報告已保存到：{report_path}")

def main():
    parser = argparse.ArgumentParser(description='Token Loss 系統比較實驗')
    parser.add_argument('--epochs', type=int, default=10, help='訓練輪數')
    parser.add_argument('--max_samples', type=int, default=100, help='最大樣本數')
    
    args = parser.parse_args()
    
    # 創建結果目錄
    os.makedirs("results", exist_ok=True)
    
    # 運行比較實驗
    results = run_training_comparison(args.epochs, args.max_samples)
    
    print("\n" + "=" * 60)
    print("Token Loss 系統比較實驗完成！")
    print(f"請查看 results/token_loss_comparison_report.md 獲取詳細結果")
    print("=" * 60)

if __name__ == "__main__":
    main()
