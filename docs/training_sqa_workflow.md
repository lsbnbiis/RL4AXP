# RL4AXP — Start Training 與 SQA Refinement 使用時機及流程

## 兩者的本質差異

| 項目 | Start Training（PPO） | SQA Refinement |
|------|----------------------|----------------|
| 演算法 | Proximal Policy Optimization（近端策略優化） | Simulated Quantum Annealing（模擬量子退火） |
| 本質 | 強化學習，從頭探索序列空間，學習「什麼樣的序列有活性」的策略 | 量子啟發式後處理，針對已有候選序列做 position-wise 微調 |
| 輸入 | Target peptide + 選定的 Reward models + Hyperparameters | PPO 訓練後找到的 top-N candidates |
| 輸出 | 訓練好的 Actor/Critic 模型權重 + 候選序列 DataFrame | 每條 candidate 的精修版本（Refined sequence）及分數比較 |
| 依賴關係 | 不依賴 SQA，可獨立執行 | **必須先完成 PPO 訓練**，有 candidates 才能執行 |
| 執行時間 | 長（數千至數萬 episodes） | 短（每條序列約 500 annealing steps） |

---

## 各元件說明

### PPO（Start Training）

訓練由三個神經網路組成：

- **Actor1**：選擇要修改哪個位置（position selection）
- **Actor2**：選擇將該位置換成哪個胺基酸（amino acid selection）
- **Critic**：評估當前序列的預期累積獎勵

每個 episode 同時跑 `N_PARALLELS`（預設 200）條平行序列，每條做 `TIME_HORIZON`（預設 5）步修改。
Buffer 累積到 `BUFFER_SIZE`（預設 10240）筆後，觸發一次 `learn()`，更新網路權重。

**Reward 組成（每步）：**
```
reward = heuristic_step
       + Σ weight[m] × direction[m] × (prob_curr[m] - prob_prev[m])
       - HEM_PENALTY_SCALE × max(0, HEM_prob - HEM_THRESHOLD)
```

**Terminal bonus（episode 結束時額外加）：**
```
reward += heuristic_final
        + Σ weight[m] × direction[m] × normalized_improvement[m]
```

### SQA（SQA Refinement）

針對 PPO 找到的候選序列，用量子退火進一步精修：

1. 從 `exp_results_df` 取 top-N 序列（依 Heuristic score 排序）
2. 用 PepBERT 提取每個位置的 attention 作為 **J_ij coupling**（QUBO 矩陣的耦合項）
3. 用訓練好的 Actor1/Actor2 選出最有潛力的突變位置和胺基酸，形成 **h_i bias**
4. 組成 QUBO 矩陣，執行 GPU 加速的 SQA（`SQA_N_TROTTER` 個 Trotter 切片，`SQA_N_STEPS` 步退火）
5. 輸出精修後的序列，並重新評分（所有 reward models）

---

## 完整使用流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 設定參數（Sidebar）                                  │
│                                                              │
│  • Target Peptide：輸入起始序列                               │
│  • Reward Models：勾選要最佳化的目標                          │
│  • Weights：調整各 model 的 reward 權重                       │
│    - AMP: 1.0  ACP: 0.6  AFP: 0.6  AVP: 0.6  MRSA: 1.0  HEM: 2.5 │
│  • HEM thr: 0.30  HEM penalty: 1.0                          │
│  • N Parallels: 200  Time Horizon: 5                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Start Training（PPO）                               │
│                                                              │
│  點擊 ▶ Start Training                                       │
│  • Status: Initializing → Training                          │
│  • Monitor tab 觀察：                                        │
│    - Cumulative Reward：應逐漸上升                           │
│    - Model Probabilities：AMP/ACP/AFP/AVP/MRSA 上升，HEM 下降 │
│    - Actor Losses：Buffer 累積到 10240 後開始出現             │
│  • 建議跑 5,000 ~ 50,000 episodes 視收斂狀況決定             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 查看 Candidates                                     │
│                                                              │
│  切換至 🧬 Candidates tab                                    │
│  • 篩選條件：Threshold 調整顯示高分序列的閾值                  │
│  • 排序：依 Cumulative Reward / Heuristic / 各 model 機率    │
│  • 匯出：Export CSV 保存候選序列                              │
│  • 觀察 HEM-Prob 是否已壓低至目標值（建議 < 0.3）             │
└──────────────────────┬──────────────────────────────────────┘
                       │
              HEM 仍偏高？  ──→  調整 HEM weight / threshold
                       │        點 Reset → 重新 Start Training
                       │ HEM 已控制
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: SQA Refinement（精修）                              │
│                                                              │
│  切換至 ⚛ SQA tab                                           │
│  • Top-N：選取前幾名序列做精修（建議 10~50）                  │
│  • N Positions：考慮的突變位置數（預設 8）                    │
│  • N AAs per position：每位置考慮的胺基酸數（預設 3）         │
│  • N Trotter slices：量子維度（預設 20）                      │
│  • N Steps：退火迭代次數（預設 500）                          │
│  點擊 Run SQA                                                │
│  → 比較 Original vs Refined 序列的各 model 分數變化          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Scorer（單序列驗證）                                 │
│                                                              │
│  切換至 🔬 Scorer tab                                        │
│  • 貼入任意序列（PPO 或 SQA 輸出）                            │
│  • 選擇要評分的 models                                       │
│  • 調整 HEM Concentration（μg/mL）                           │
│  • 點擊 Score → 查看各 model 機率分數及 Heuristic score       │
└─────────────────────────────────────────────────────────────┘
```

---

## 何時重跑 Training？

| 情況 | 建議動作 |
|------|---------|
| HEM 持續偏高（> 0.4） | 提高 HEM weight（→ 3.0~4.0）、降低 HEM thr（→ 0.2）後 Reset + 重跑 |
| AMP/活性分數停滯 | 檢查 Target peptide 是否過難；適當提高 AMP weight |
| Reward 曲線不收斂 | 降低 N Parallels、降低 LR（config.py）；或換 Target peptide |
| 換了新的 Target peptide | 必須 Reset 後重跑，舊的 buffer 與序列不適用 |
| 已有好的 candidates 但想進一步優化 | 不需重跑，直接做 SQA Refinement |

---

## 關鍵參數對照（config.py）

```python
N_PARALLELS     = 200        # 平行序列數；越大探索越廣，記憶體需求越高
TIME_HORIZON    = 5          # 每 episode 修改步數
BUFFER_SIZE     = 2048 * TIME_HORIZON  # 觸發 learn() 的 buffer 大小
AGENTS_LR       = 2e-5       # 學習率

REWARD_WEIGHTS  = {          # 各 model reward 權重（可在 Dashboard 調整）
    "AMP": 1.0, "ACP": 0.6,
    "AFP": 0.6, "AVP": 0.6,
    "MRSA": 1.0, "HEM": 2.5,
}
HEM_THRESHOLD     = 0.3      # 超過此值開始額外懲罰
HEM_PENALTY_SCALE = 1.0      # 懲罰強度係數

USE_SQA         = True       # 是否在 checkpoint 時自動觸發 SQA
SQA_N_TROTTER   = 20
SQA_N_STEPS     = 500
SQA_N_POSITIONS = 8
SQA_N_AAS       = 3
```

---

*文件產生日期：2026-06-10*
*系統：RL4AXP — AMP Peptide Design（PPO + SQA，RTX 6000 Ada）*
