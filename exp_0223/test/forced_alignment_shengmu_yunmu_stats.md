# Forced Alignment：聲母/韻母統計（Noisy 路徑）

比較：`noisy_through_teacher`（VQ） vs `noisy_through_teacher_no_vq`（No-VQ）
差值定義：`No-VQ LSE - VQ LSE`，負值代表 No-VQ 較好。

## Token-level 平均差（forced alignment）

| 類別 | N tokens | HF 平均差 | 95% CI | Wilcoxon p | t-test p |
|---|---:|---:|---|---:|---:|
| 聲母 (shengmu) | 28 | 0.000506 | [-0.017392, 0.019242] | 8.489e-01 | 9.591e-01 |
| 韻母 (yunmu) | 55 | -0.010492 | [-0.024729, 0.003776] | 1.568e-01 | 1.625e-01 |

## 平均差摘要

- HF 聲母平均差：`0.000506`
- HF 韻母平均差：`-0.010492`
- Full-band 聲母平均差：`0.005824`
- Full-band 韻母平均差：`0.001641`

## 對齊信心（每句平均 token score）

- sample01: mean=0.978, min=0.703, max=0.999
- sample02: mean=0.987, min=0.863, max=0.999
- sample03: mean=0.927, min=0.127, max=0.998

## 注意
- 這裡的子音/母音以中文語音學常用的「聲母/韻母」近似，而非 IPA phone 集。
- 樣本數目前僅 3 句（token 級統計），結論屬初步證據。
