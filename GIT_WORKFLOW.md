# Git 版本控制工作流程指南

## 🌟 當前分支策略

### 主要分支
- **main**: 穩定的主分支，包含基礎 TTT2 實現
- **fix-ttt2-residual-block-and-manifold**: 修復和改進分支（當前工作分支）

## 🔄 建議的工作流程

### 1. 日常開發流程
```bash
# 確保在 fix 分支上工作
git checkout fix-ttt2-residual-block-and-manifold

# 提交當前更改
git add .
git commit -m "實驗編號_YYYYMMDD: 描述更改內容"

# 推送到遠程倉庫
git push origin fix-ttt2-residual-block-and-manifold
```

### 2. 使用 Worktree 同時管理兩個分支
```bash
# 在上級目錄創建 main 分支的工作樹
cd /home/sbplab/ruizi
git worktree add WavTokenize-main main

# 現在您有兩個工作目錄：
# WavTokenize/          - fix 分支（當前開發）
# WavTokenize-main/     - main 分支（穩定版本）
```

### 3. 實驗提交格式
每次實驗提交應包含：
```
實驗編號_日期: 實驗目的

- 實驗背景：
- 動機：
- 目的：
- 預期結果：
- 實際結果：
- 結果解讀：
- 下一步計劃：
- 重現步驟：
```

## 📊 分支比較和合併策略

### 查看分支差異
```bash
# 查看 fix 分支相對於 main 的差異
git diff main..fix-ttt2-residual-block-and-manifold

# 查看檔案變更統計
git diff --stat main..fix-ttt2-residual-block-and-manifold
```

### 選擇性合併
```bash
# 如果需要將特定改進合併到 main
git checkout main
git cherry-pick <commit-hash>

# 或者合併整個分支（謹慎使用）
git merge fix-ttt2-residual-block-and-manifold
```

## 🔧 Worktree 管理命令

### 查看所有 worktree
```bash
git worktree list
```

### 刪除 worktree
```bash
# 先刪除目錄內容
rm -rf /path/to/worktree

# 然後清理 git 記錄
git worktree prune
```

### 在不同 worktree 間切換
```bash
# 去 main 分支工作目錄
cd /home/sbplab/ruizi/WavTokenize-main

# 回到 fix 分支工作目錄  
cd /home/sbplab/ruizi/WavTokenize
```

## 🎯 您的具體使用建議

1. **保持當前結構**: 繼續在 `/home/sbplab/ruizi/WavTokenize/` 目錄的 `fix-ttt2-residual-block-and-manifold` 分支上工作

2. **可選創建 main worktree**: 如果需要比較或參考 main 分支，可以創建 worktree

3. **實驗管理**: 每個重要實驗都應該有對應的 commit，並更新 REPORT.md

4. **檔案管理**: 使用 .gitignore 忽略臨時檔案，使用 Git LFS 管理大檔案

## 📝 實驗提交範例

```bash
# 實驗完成後
git add test_ttt2_outside.py REPORT.md
git commit -m "TTT2_20250814: 修復 outside 音檔測試能量損失問題

- 實驗背景: test_ttt2_outside.py 生成的音檔能量大幅降低，缺乏人聲
- 動機: 修復音檔生成管道，確保與 ttt2.py 一致的輸出品質  
- 目的: 實現高品質的 outside 音檔測試系統
- 預期結果: 生成與 ttt2.py 相當能量的音檔
- 實際結果: [待填入]
- 結果解讀: [待填入]
- 重現步驟: python test_ttt2_outside.py --checkpoint output4/best_model.pth"
```
