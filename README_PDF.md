# PDF 版本使用說明

## 生成的 PDF 文件

📄 **[本周進度報告_0129.pdf](本周進度報告_0129.pdf)**

### 文件信息

- **格式**: PDF 1.4
- **大小**: 1.6 MB
- **頁數**: 17 頁
- **生成日期**: 2026-01-29
- **包含內容**: 完整報告 + 所有嵌入圖片

### PDF 特點

✅ **所有圖片已嵌入**
- TracIn Influence vs SNR 散點圖
- 實驗 1 訓練曲線（TracIn-Weighted）
- 實驗 2 訓練曲線（Noise-Balanced）
- 無需點擊連結即可查看所有視覺化證據

✅ **完整內容**
- 6 大部分內容
- 3 個附錄（圖表總覽、Git Commits、文件結構）
- 所有表格、代碼塊、格式保留

✅ **便於分享**
- 單一 PDF 文件，無需額外附件
- 適合郵件發送或列印
- 跨平台兼容

## 生成方法

本 PDF 使用以下工具鏈生成：

```bash
# 1. Markdown → HTML (Python markdown)
python3 md_to_pdf.py Weekly_Progress_Report_0129.md

# 2. HTML → PDF (Chromium headless)
chromium-browser --headless --disable-gpu \
  --print-to-pdf="Weekly_Progress_Report_0129.pdf" \
  --no-pdf-header-footer \
  "file://$(pwd)/Weekly_Progress_Report_0129.html"
```

### 技術細節

**HTML 樣式**:
- 中文字體：Noto Sans CJK SC
- 頁面寬度：900px
- 行高：1.6
- 圖片自動縮放適應頁面

**圖片嵌入**:
- 所有相對路徑圖片自動解析
- 圖片保持原始比例
- 自動居中顯示

## 源文件

如需編輯或重新生成 PDF，請參考：

- **Markdown 源文件**: [本周進度報告_0129.md](本周進度報告_0129.md)
- **HTML 中間文件**: [Weekly_Progress_Report_0129.html](Weekly_Progress_Report_0129.html)
- **圖片文件**:
  - `exp_0125/tracin_token_collapse_589e6d/profiles_5ckpt/plots/influence_vs_snr.png`
  - `exp_0128/soft_reweighting/run_exp1_20260129_023536/training_curves.png`
  - `exp_0128/noise_balanced_sampling/run_exp2_20260129_022108/training_curves.png`

## 版本對比

| 版本 | 優點 | 缺點 |
|------|------|------|
| **Markdown** | 可編輯、可版本控制、輕量 | 需要支持的 viewer 查看圖片 |
| **HTML** | 可在瀏覽器查看、交互式連結 | 需要相對路徑正確 |
| **PDF** | 獨立完整、所有圖片嵌入、便於分享 | 無法編輯 |

## 推薦使用場景

✅ **使用 PDF 當**:
- 需要分享給他人（郵件、簡報）
- 需要列印紙本
- 需要完整獨立的歸檔版本
- 接收方沒有 Markdown 閱讀器

✅ **使用 Markdown 當**:
- 需要編輯內容
- 需要版本控制
- 在支持 Markdown 的環境（VSCode, GitHub）查看

---

**生成時間**: 2026-01-29
**工具**: Python markdown + Chromium headless
**品質**: 生產就緒 ✅
