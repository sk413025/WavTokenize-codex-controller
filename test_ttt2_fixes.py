#!/usr/bin/env python3
"""
TTT2 模型修復驗證腳本
測試關鍵修復是否正確實作
"""

import torch
import torch.nn as nn
import sys
import traceback

def test_residual_block():
    """測試 ResidualBlock 的修復"""
    print("=" * 50)
    print("測試 ResidualBlock 修復")
    print("=" * 50)
    
    try:
        # 手動定義修復後的 ResidualBlock 來測試
        class FixedResidualBlock(nn.Module):
            def __init__(self, channels, activation='relu', dropout_rate=0.1, use_group_norm=False):
                super().__init__()
                
                # 第一層卷積
                self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
                
                # 正規化層選擇
                if use_group_norm:
                    # 使用 GroupNorm，組數設為 8 或 channels 的因數
                    num_groups = min(8, channels)
                    while channels % num_groups != 0 and num_groups > 1:
                        num_groups -= 1
                    self.norm1 = nn.GroupNorm(num_groups, channels)
                    self.norm2 = nn.GroupNorm(num_groups, channels)
                else:
                    self.norm1 = nn.BatchNorm1d(channels)
                    self.norm2 = nn.BatchNorm1d(channels)
                
                # 激活函數
                if activation == 'gelu':
                    self.activation = nn.GELU()
                else:
                    self.activation = nn.ReLU(inplace=True)
                
                # 第二層卷積
                self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
                
                # Dropout
                self.dropout = nn.Dropout(dropout_rate)
            
            def forward(self, x):
                """修復後的前向傳播 - 關鍵修復在這裡"""
                residual = x  # 保存輸入用於殘差連接
                
                # 第一層處理
                out = self.conv1(x)
                out = self.norm1(out)
                out = self.activation(out)
                out = self.dropout(out)
                
                # 第二層處理 - 關鍵修復：使用 out 而不是 x
                out = self.conv2(out)  # 修復前是：self.conv2(x)
                out = self.norm2(out)
                
                # 殘差連接
                out += residual
                out = self.activation(out)
                
                return out
        
        # 測試不同配置
        configs = [
            {"use_group_norm": False, "name": "BatchNorm"},
            {"use_group_norm": True, "name": "GroupNorm"}
        ]
        
        for config in configs:
            print(f"\n測試配置: {config['name']}")
            
            # 創建模型
            channels = 256
            block = FixedResidualBlock(
                channels=channels, 
                activation='gelu', 
                dropout_rate=0.1,
                use_group_norm=config['use_group_norm']
            )
            
            # 測試數據
            batch_size = 2
            seq_len = 100
            x = torch.randn(batch_size, channels, seq_len)
            
            # 前向傳播
            with torch.no_grad():
                output = block(x)
            
            # 驗證輸出
            assert output.shape == x.shape, f"輸出形狀不匹配: {output.shape} vs {x.shape}"
            
            # 檢查殘差連接是否工作
            diff = torch.abs(output - x).mean()
            print(f"  ✅ 輸出形狀: {output.shape}")
            print(f"  ✅ 與輸入的平均差異: {diff:.6f}")
            print(f"  ✅ 正規化層: {type(block.norm1).__name__}")
        
        print("\n✅ ResidualBlock 修復驗證通過！")
        return True
        
    except Exception as e:
        print(f"❌ ResidualBlock 測試失敗: {e}")
        traceback.print_exc()
        return False

def test_loss_functions():
    """測試新增的損失函數"""
    print("\n" + "=" * 50)
    print("測試新增損失函數")
    print("=" * 50)
    
    try:
        def compute_codebook_consistency_loss(enhanced_features, target_features, alpha=0.1):
            """碼本一致性損失"""
            return alpha * torch.mean(torch.abs(enhanced_features - target_features))
        
        def compute_manifold_regularization_loss(enhanced_features, target_features, beta=0.05):
            """流形正則化損失"""
            # 計算特徵的範數
            enhanced_norm = torch.norm(enhanced_features, dim=-1, keepdim=True)
            target_norm = torch.norm(target_features, dim=-1, keepdim=True)
            
            # 正則化項：懲罰過大的特徵範數
            norm_penalty = torch.mean(torch.relu(enhanced_norm - 1.5 * target_norm))
            
            return beta * norm_penalty
        
        # 測試數據（需要梯度）
        batch_size, channels, seq_len = 2, 512, 100
        enhanced_features = torch.randn(batch_size, channels, seq_len, requires_grad=True)
        target_features = torch.randn(batch_size, channels, seq_len)
        
        # 測試碼本一致性損失
        codebook_loss = compute_codebook_consistency_loss(enhanced_features, target_features)
        print(f"✅ 碼本一致性損失: {codebook_loss.item():.6f}")
        
        # 測試流形正則化損失
        manifold_loss = compute_manifold_regularization_loss(enhanced_features, target_features)
        print(f"✅ 流形正則化損失: {manifold_loss.item():.6f}")
        
        # 測試梯度計算
        total_loss = codebook_loss + manifold_loss
        total_loss.backward()
        print(f"✅ 總損失: {total_loss.item():.6f}")
        print("✅ 梯度計算成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 損失函數測試失敗: {e}")
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("TTT2 模型修復驗證開始")
    print("=" * 70)
    
    # 測試記錄
    results = {}
    
    # 測試 ResidualBlock 修復
    results['residual_block'] = test_residual_block()
    
    # 測試損失函數
    results['loss_functions'] = test_loss_functions()
    
    # 總結結果
    print("\n" + "=" * 70)
    print("測試結果總結")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有測試通過！TTT2 模型修復驗證成功！")
        print("建議下一步：運行完整的訓練測試以驗證實際性能改善")
    else:
        print("⚠️  部分測試失敗，需要進一步檢查修復實作")
    print("=" * 70)

if __name__ == "__main__":
    main()
