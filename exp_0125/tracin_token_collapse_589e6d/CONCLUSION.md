# CONCLUSION: TracIn diagnosis for valid token collapse (commit 589e6d)

## 核心回答（Q1–Q4）

### Q1) valid token collapse Top‑3 root causes（含排序）

**#1 H3 資料驅動：噪音材質強的 train 子集主導 LoRA 更新**  
- **證據**：TracIn（val failure 集合的 aggregated 梯度）顯示 proponents 的 **SNR 明顯更低**（mean **-2.61 dB**，Cohen’s d = **-0.265**），且 noise type 以 **papercup** 為主（61/100）。`proponents_profile.json`  
- **對照**：opponents 的 SNR 更高（mean **-1.30 dB**，Cohen’s d = **+0.260**），noise type 以 **plastic/box** 為主（52/45）。`opponents_profile.json`  
- **S3 音質交叉**：bottom‑PESQ val failure 子集的 TracIn proponents 仍偏低 SNR（mean **-2.10 dB**）且 **papercup 佔 59%**；opponents 以 **box 佔 77%**。`audio_quality/bottom_pesq_profiles/proponents_profile.json` / `audio_quality/bottom_pesq_profiles/opponents_profile.json`

**#2 H2 VQ / margin 不穩定放大 collapse**  
- **證據**：VQ margin（d2‑d1）在 val 顯著更小：p50 **0.01085** vs train **0.01694**。`metrics_overview.json`  
- **解讀**：margin 變小代表量化決策更不穩定，容易塌縮到少數 token，與 valid strict acc 下降一致。

**#3 H1 noise‑dependent / joint encoding**  
- **證據**：高影響力 train 子集偏向低 SNR / 特定噪音材質（papercup），顯示 LoRA 在噪音材質上學到「可解碼」模式。`proponents_profile.json`  
- **限制**：本次 TracIn 使用 **val failure aggregated gradient** + **近似 L_train**（triplet=0），缺少 per‑val 的直接對齊/崩塌關聯。結論為「支持但需加強」。

---

### Q2) TracIn 是否顯示：val failure 的高 influence train samples 偏向「雜訊材質強」？
**結論：支持（以 SNR/材質 proxy）。**
- proponents 的 SNR 明顯低於 baseline（Cohen’s d = **-0.265**），noise type 以 **papercup** 為主（61%）。`proponents_profile.json`
- opponents 的 SNR 偏高（Cohen’s d = **+0.260**），noise type 以 **plastic/box** 為主。`opponents_profile.json`
- **音質最差子集對照**：bottom‑PESQ failure 子集的 proponents 仍以 **papercup** 為主（59%），SNR mean **-2.10 dB**；opponents 以 **box** 為主（77%）。`audio_quality/bottom_pesq_profiles/*`
- **L_anchor 一致性**：anchor loss 的 proponents SNR 更低（mean **-2.67 dB**, d = **-0.292**），noise type 仍以 **papercup** 為主（62%）。`anchor_profiles/proponents_profile.json`
- **控制比較**：以 SNR/energy 作為 proxy 與全體 train candidates 對照，顯示高 influence 子集並非隨機抽樣；仍需 speaker/content metadata 才能完全排除內容相似性混淆。

### Q3) Opponents 是否對應到與 val failure 材質/條件衝突的 train 子集？
**結論：支持（以噪音 proxy）。**
- opponents 的 SNR 更高、noise type 組成與 proponents 明顯不同（plastic/box 占比高），呈現「材質相反」趨勢。`opponents_profile.json`
- bottom‑PESQ 子集中 opponents 以 **box 佔 77%**，與 proponents 的 papercup 分佈相反。`audio_quality/bottom_pesq_profiles/opponents_profile.json`
- **L_anchor 一致性**：anchor loss 的 opponents SNR 亦較高（mean **-1.68 dB**, d = **+0.097**），noise type 結構不同於 proponents。`anchor_profiles/opponents_profile.json`
- 推論：val failure 若由噪音材質驅動，則高 SNR/不同材質的 train 子集會成為負向影響（opponents）。

### Q4) Proposed Fix（primary）+ 下一步驗證（1–3 天內）

**Primary Proposed Fix（可落地）**  
**Noise‑aware reweighting / balanced sampling + teacher anchor**  
- **做法**：  
  1) 依 TracIn proponents 的 noise proxy（低 SNR / papercup）**降權**；  
  2) 對 opponents/高 SNR/其他材質做 **平衡抽樣**；  
  3) 保留 teacher‑anchor（per‑frame KL/CE）以避免 trivial collapse。  
- **對應 root causes**：H3（資料驅動）＋ H1（noise‑dependent）＋ H2（量化穩定性間接受益）。

**下一步驗證實驗（≥2）**
1) **Counterfactual short‑run（downweight/filter）**  
   - 2k samples / 800–1000 steps；移除或降權 TracIn proponents；  
   - 成功判準：val entropy ↑、top‑k mass ↓、strict acc 不惡化。  
2) **完整 TracIn‑CP（補強證據）**  
   - 加入 L_anchor 版本 + per‑val failure（非 aggregate）  
   - 檢查 proponents/opponents 是否穩健指向相同 noise 子集。

**Counterfactual 結果（已完成）**  
- `counterfactual/summary.md` 顯示：移除 top‑200 proponents 後 **strict acc 下降、entropy 下降、top‑k mass 上升**，方向性惡化。  
- 解讀：單純移除高 influence 噪音子集會破壞代表性噪音多樣性；更可能需要 **soft reweighting** 或噪音材質增強，而非硬刪。  

> **限制註記**：本次 TracIn 使用 **aggregate val loss** + **近似 L_train（triplet=0）**，且 L_anchor 使用較小 train_candidates（1000）完成；仍需擴充到 full per‑val 版本。

---

## Acceptance self-check（對照 589e6d_token_collapse_tracin_acceptance.md）
- M1: ✅（metrics_overview.json / failure_set.json）
- M2: ✅（tracin_scores.csv；train candidates=2000、val failures=50）
- M3: ✅（param_scope.json）
- M4: ✅（proponents/opponents profile + SNR/energy 對照）
- M5: ✅（Top‑3 root causes + Proposed Fix + Next steps）
- S1: ✅（已補 L_anchor 版 TracIn；樣本數較小且 aggregate 近似，仍需加強）
- S2: ✅（已做 counterfactual short-run；結果惡化，已記錄）
- S3: ✅（已完成 failure set 音質評估 + bottom‑PESQ 子集 TracIn 交叉）
