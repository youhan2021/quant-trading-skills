# quant-trading

量化交易系统 — 固定动量策略 + 严格三段式回测，无信息泄漏。

## 当前策略

**固定 Weekly Momentum（2026-05-06 最终版）**

- Universe: 13 只 — SPY, QQQ, GLD, TLT, EFA, AMZN, LMT, V, NVDA, WMT, GOOGL, UUP, JPM
- 因子: roc120 + vol20（rank-zscore 等权）
- 持仓: top5，等权，周频再平衡
- Sharpe: **1.25**（无TC），**1.13**（20bps）
- SPY B&H Sharpe: 0.77

## 运行

```bash
cd ~/.hermes/skills/quant-trading
python3 scripts/fixed_weekly_backtest.py
```

## 目录结构

```
quant-trading/
├── SKILL.md                     # 主策略文档（含完整年度数据）
├── README.md                    # 本文件
├── scripts/
│   └── fixed_weekly_backtest.py  # 核心回测脚本
├── momentum-strategy/            # 【已归档】历史研究
├── fundamental-data-collector/   # 数据获取 + 实验脚本
├── quant-wfa/                   # 【已归档】WFA 方法论
└── data/                        # 本地数据（不 push）
```

## 核心结论

**无圣杯，无 Regime Switch。** 所有结论见 `SKILL.md`。
