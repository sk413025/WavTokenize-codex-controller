# 資料集對齊問題梳理與快速驗證報告

更新時間：2025-12-12  
相關訓練版本：`exp_1210`（修復 EMA 漂移後的 LoRA 訓練）

本文件是為了回答一個實際訓練卡關的問題：在修復 EMA 造成的 codebook 漂移後，token exact accuracy 確實能上升，但速度非常慢且很早 plateau。  
要判斷下一步該改 loss、LR、LoRA 容量之前，我們必須先確定 noisy/clean 的監督資料本身在時間軸上是否可靠；否則任何「token 要對齊」的訓練/評估都可能變成徒勞。

---

## 前言：為什麼要提這些假設？

我們看到的現象可以來自兩大類原因：

```
                 token acc 慢 / plateau
                           |
        +------------------+------------------+
        |                                     |
   (A) 資料層：noisy/clean              (B) 模型層：任務本來就硬
       對齊或切段有問題                    /離散監督不足
        |                                     |
  -> exact token match 會被              -> 需要調 loss/LR
     錯位 frame 污染                         /容量/離散目標
```

如果是 (A)，那麼你再怎麼調模型都只是在追一個被污染的目標；  
如果 (B)，資料是對的，那就回到模型/目標設計去加速跨 VQ 邊界。

因此我先提出三個「可快速被驗證或推翻」的假設（H1–H3），用最少的統計把 (A) 與 (B) 分開：
- H1：noisy/clean 內容時間軸同步嗎？（若不同步，token match 無意義）
- H2：pair 長度/切段一致嗎？batch padding 會不會把監督弄髒？
- H3：cache 內的 tokens 是否已經對齊？（用來定位 mismatch 來源）

接下來的章節就是依序驗證這些假設、量化嚴重程度，最後給出改善順序。

## 現象：為什麼 train/val token acc 看起來都很慢？

在 EMA 漂移修復後，實驗曲線呈現一致的特徵：
- train token acc 能從 ~26% 漲到 ~30%，但前 5–10 epoch 漲一段後就變得很平。
- val token acc 也同步上升，但更早 plateau，且 train‑val gap 明顯。

直覺上會覺得「如果 TRAIN 的 noisy/clean 很同步，train acc 應該很快衝上去」，但這裡要先釐清一件事：  
**train/val 指標的計算方式與資料幾何會讓 acc 天生看起來慢，即使 TRAIN 本身沒有大尺度不同步。**

兩個最常被混在一起的來源如下：

```
來源 1：per‑pair mismatch（同一對 noisy/clean 長度不同）
  noisy : |====================|
  clean : |====================|=====tail=====|
                              ^ 尾段錯位，監督變髒

來源 2：cross‑sample mismatch（batch 內不同樣本長度差很大）
  sample A noisy/clean : |===========|
  sample B noisy/clean : |=========================|
  collate_fn 會 pad 到 batch max_len
  -> 尾段大量 padding frame 仍被算入 acc/loss
```

我們接下來的假設與驗證，就是要分清楚：
1. train/val acc 慢是不是來自來源 1 的「資料對齊/切段問題」；
2. 如果資料是可靠的，慢是否主要是任務幾何/目標設計所致。

## 0. 背景：目前訓練資料流長什麼樣？

cache 檔案是索引/metadata（只存 path 與 tokens），訓練時再去讀 wav：

```
TRAIN_CACHE/VAL_CACHE  (noisy_path, clean_path, noisy_tokens, clean_tokens, ...)
          |
          v
NoisyCleanPairDataset.__getitem__()
  - cache 沒有 waveform -> 走 noisy_path/clean_path 讀 wav
          |
          v
collate_fn()
  - 取 batch 內 noisy 的 max_len
  - noisy/clean 各自 truncate/pad 到 max_len
          |
          v
model(noisy_wav, clean_wav) -> teacher_codes(clean), student_codes(noisy)
loss(feature + triplet)
```

關鍵：**目前沒有 per-pair 的對齊（min_len 截斷或 mask）**，只有「每個 batch 內各自 padding 到同一長度」。

---

## 1. 假設（Hypotheses）

H1. **noisy/clean 時間軸內容其實是同步的（沒有系統性 delay / drift）。**  
若不成立，exact token match 本身就無意義。

H2. **noisy/clean pair 的長度在某些 split 會不一致，且 padding 方式會污染監督與 metric。**  
若嚴重，batched 下的 token acc/feature loss 會被尾段錯位資料拖累。

H3. **（輔助）cache 內 tokens 是否已對齊？**  
若 tokens 長度一致，代表你產 tokens 時曾做過對齊；但訓練用 wav 時如果沒同步，會重新引入 mismatch。

---

## 2. 快速檢查方式與結果

### 2.1 檢查 cache 內容（驗證 H3）

**方法**：直接讀 `TRAIN_CACHE/VAL_CACHE` 的 sample keys 與 token 長度。

**結果**：
- cache 只有 `noisy_path/clean_path` 與 `noisy_tokens/clean_tokens`，**沒有 waveform**。
- `noisy_tokens` 與 `clean_tokens` 的長度在 val 內 **100% 完全一致**（token length diff = 0）。  
  → 代表 tokens 生成流程應該做過 per-pair 對齊/截斷。

### 2.2 檢查 wav pair 長度差分佈（驗證 H2）

**方法**：
- 用 `torchaudio.info()` 讀 noisy/clean wav 的 `num_frames` 與 `sample_rate`；
- 全部換算到 24kHz 後比較長度差。

**VAL（2592 對）結果：嚴重**
- 有長度不一致的 pair：**40.35%**
- clean 與 noisy 長度差（abs）：
  - mean：**6394 samples ≈ 0.266s**
  - median：0（表示是「部分 pair 大幅 mismatch」）
  - p90：22506 samples ≈ 0.938s
  - p95：27939 samples ≈ 1.164s
  - p99：35984 samples ≈ 1.499s
  - max：57948 samples ≈ 2.414s
- 方向性：**所有 mismatch 都是 clean 比 noisy 長**  
  - noisy longer 0%，clean longer 40.35%

**TRAIN（抽 2000/14400 對）結果：基本不嚴重**
- header 層面 diff!=0 很多，但幾乎都是 1–2 samples 的微差。
- meaningful mismatch 很少：
  - diff > 0.01s：2.95%
  - diff > 0.1s：2.6%
  - diff > 0.5s：0.8%
  - max：29612 samples ≈ 1.234s（極少數 outlier）

**split/群組性**
- VAL 的 mismatch 不是平均分布：  
  - `boy7` mismatch 57%（mean clean-noisy ≈ +0.376s）  
  - `boy8` mismatch 42%（mean ≈ +0.264s）  
  - `girl9` mismatch 17%（mean ≈ +0.122s）  
  - 多個 `content_id` 系統性 clean 多 0.65–0.99s，且該句子 66.7% 样本 mismatch。

**推論**：這更像是 **VAL 切段/後處理規則不同**（如 clean 沒做同樣 trimming/VAD），不是錄音本身不同步。

### 2.3 檢查時間軸對齊（驗證 H1）

**方法**：
1. 對每一對 noisy/clean（先截到 min_len 的 1.5 秒），用 teacher encoder 產生 frame 特徵。
2. 做 ±6 frame（約 ±150ms）shift 掃描，找最大 cosine 相似度的最佳 shift。
3. 看最佳 shift 分佈與 token acc 是否因 shift 有提升。

**結果（VAL 抽 50 對）**：
- 最佳 shift 分佈：
  - mean 0.0 frame，std 0.8 frame
  - p95(|shift|)=2 frames
  - 66% sample 最佳 shift=0
  - |shift|>=3 frames 的 sample = 0%
- token acc（teacher noisy vs clean）  
  - 不 shift：mean 38.12%  
  - 套最佳 shift 後：mean 38.11%（幾乎不變）

**結論**：noisy/clean 的**內容時間軸對齊良好**，沒有系統性 delay；exact token match 是有意義的。

### 2.4 padding 污染的直觀證據（支持 H2）

同一份 VAL：

```
Case A: per-pair 先對齊 (min_len)

noisy : |====================|
clean : |====================|
           ^ 同步、可比

teacher(noisy) vs teacher(clean) token acc  ~35–38%


Case B: 現況 batched padding（clean 常更長）

noisy : |====================|----0----0----0----|
clean : |====================|========tail=======|
                               ^ 尾段錯位

batched 下 teacher(noisy) vs teacher(clean) token acc  ~3–5%
```

→ 現況的 batch padding 會把一大段「clean tail」拿去跟 noisy 的 padding/別段內容對比，使 acc/loss 被大幅壓低。

---

## 3. 資料問題對訓練的影響（為何 acc 升得慢）

這一節把「TRAIN mismatch 不嚴重」與「train acc 仍很慢」放在同一張因果圖中解釋。

### 3.1 指標/監督被 batch padding 稀釋

即使 TRAIN 的 per‑pair 大 mismatch 很少，batch 內仍有大量 padding frame（來源 2），原因是不同 utterance 長度本來就差很多。  
padding frame 會讓：
- **acc 被低估**：teacher/student 在 padding 上輸出的 codes 幾乎隨機或固定，拉低平均。
- **梯度變吵**：loss 在這些沒有對齊意義的 frame 上仍被計算，等於加了噪聲監督。

VAL 更嚴重是因為還疊了來源 1（clean 系統性更長）。我們用 teacher 做過對照：
- per‑sample min_len 對齊下：teacher(noisy vs clean) acc ~35–38%
- 現況 batched/padded 下：acc 只剩 ~3–5%  
→ 這說明「batched 指標」可以被 padding 壓到失真，因此不能用它來判斷模型是否在有效學習。

### 3.2 noisy→clean 的資料上限本來就不高

上面 min_len 的 teacher acc (~35–38%) 其實就是 noisy/clean 在離散空間的可達上限之一。  
你現在 train acc ~30% 已經接近這個天花板區間，所以後續增益自然會變慢、曲線會提前變平。

### 3.3 連續特徵幾何讓「跨 VQ 邊界」天然慢

我們量到：
- teacher noisy‑clean feature L2(mean) ≈ 5.37  
- clean feature 到最近 codebook L2(mean) ≈ 0.53  
- ratio ≈ 9–10x  
表示 noisy 特徵平均要跨過很多個 codebook cell 才會變成 clean token。

而目前 loss 只對連續空間（feature + triplet），沒有直接推 token（DW/soft‑CE 為 0）。  
因此 token acc 只能靠「特徵慢慢靠近、偶爾跨過邊界」來上升，曲線會呈現：
```
acc
 ^        ____----____----____  (邊界跨越是階梯/plateau 形)
 |   ____/
 |__/
 +------------------> epoch
```
這是任務本質 + 目標設計共同造成的慢，不是 TRAIN 不同步的證據。

### 3.4 小結：為什麼「TRAIN mismatch 低」仍會慢？

綜合 3.1–3.3：
- TRAIN 主要慢在來源 2（batch padding）+ 3.2/3.3（上限與幾何），不是來源 1。
- VAL 會再被來源 1（clean 長尾）額外拖慢與低估。

---

## 4. 建議的改善方案（優先順序）

### 4.1 立刻修正 per-pair 對齊（最高優先）

**方案 A：在 dataset 內先截到 min_len**

```
__getitem__:
  noisy, clean = load()
  L = min(len(noisy), len(clean))
  return noisy[:L], clean[:L]
```

**方案 B：保留原長度，但回傳 mask**

```
collate_fn:
  max_len = ...
  for each pair:
     L = min(len(noisy), len(clean))
     pad to max_len
     mask[0:L]=1, mask[L:]=0
loss/metrics only on mask==1
```

兩者都可；A 最簡單，B 對可變長更通用。

### 4.2 追查 VAL 切段流程（中優先）

因為 VAL 的 mismatch 系統性「clean 更長」，建議回頭檢查：
- VAL split 是否用不同的 VAD/trim 設定？
- clean/noisy 檔名配對是否有 off-by-one 或段落切錯？
- 是否能改用 cache 內已對齊的 tokens 或重新生成對齊後的 val wav cache。

### 4.3 更新評估方式

- 報告 token acc 時，優先用「per-sample min_len」或 masked acc。  
- 同時增加 top‑k / distance‑based 指標，避免 exact cell 邊界過硬造成誤判。

### 4.4 對齊修正後再做離散監督實驗

在監督乾淨後，再測：
- DW loss（小權重 + ramp）是否加速跨邊界；
- Soft‑CE（小權重 + 高溫 + ramp）是否有額外收益。

---

## 5. 快速再現用的指令片段（不需存檔）

**(1) 全量掃 VAL mismatch**

```bash
python - <<'PY'
import torchaudio, torch, numpy as np
from pathlib import Path
from exp_1201.config import VAL_CACHE

def resolve(p):
    p=Path(p)
    if p.is_absolute() and p.exists(): return p
    base=Path("/home/sbplab/ruizi/c_code/data")
    fn=p.name
    if "_clean_" in fn: return base/"clean/box2"/fn
    if "_box_" in fn: return base/"raw/box"/fn
    if "_papercup_" in fn: return base/"raw/papercup"/fn
    if "_plastic_" in fn: return base/"raw/plastic"/fn
    return Path(VAL_CACHE).parent.parent/fn

samples=torch.load(VAL_CACHE, weights_only=False)
diffs=[]
for s in samples:
    ni=torchaudio.info(str(resolve(s["noisy_path"])))
    ci=torchaudio.info(str(resolve(s["clean_path"])))
    ln=ni.num_frames; lc=ci.num_frames
    diffs.append(abs(ln-lc))
diffs=np.array(diffs)
sr=24000
print("pct mismatch", (diffs>0).mean()*100)
print("mean sec", diffs.mean()/sr, "p95 sec", np.percentile(diffs,95)/sr, "max sec", diffs.max()/sr)
PY
```

**(2) 對齊品質 shift 掃描（±6 frames）**

見本文件 2.3 的描述，可再依需求擴大 sample 數量。

---

## 6. 總結

- noisy/clean **內容時間軸對齊良好**（delay < 2 frames for 95% pairs）。  
- **VAL split 有大量且系統性的長度 mismatch（clean 更長）**，現況 padding 會嚴重污染監督/metric，導致 token acc 看起來極低、學習變慢。  
- 先修正 per‑pair 對齊後，再談 DW/soft‑CE 等離散監督的加速效果，才有可靠結論。
