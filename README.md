# ARSI — Autopoietic Recursive Self-Improvement

自创生递归自我改进系统。融合 Autopoiesis、MetaRSI 与 Mental World Modeling 的统一架构。

## 核心理念

> 一个自创生系统，它的改进回路本身必须是自创生生产的；而它对自身联合世界状态（物理+心智）的建模质量，决定了这个自我生产过程的质量。

## 架构

```
┌─────────────────────────────────────────────┐
│  不可变层（G1-G10 铁律，真实执行）              │
├─────────────────────────────────────────────┤
│  自创生 Governor                             │
│  三层决策：预演 → LLM → 启发式                │
│  输入: S_t = (Φ_t, Ψ_t) · 输出: a = (a_phy, a_ment) │
├──────────┬──────────┬───────────────────────┤
│ 赋能引擎  │ SIWM     │ 梦境管道               │
│ 6/9维度   │ 世界模型  │ LLM belief 调和        │
│ 完整闭环  │ L1+L2+L3 │ 自我模型刷新           │
├──────────┴──────────┴───────────────────────┤
│  Mnemosyne 统一记忆底座                       │
│  SELF / EXPERIENCE / PROXY 三区隔离           │
│  语义检索 · 跨agent共享 · 多巴胺门控           │
├─────────────────────────────────────────────┤
│  密封评估器（15任务，真实代码执行）             │
│  增益三分解 · 成本四账本                       │
└─────────────────────────────────────────────┘
```

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 设置环境变量
export ARSI_API_KEY="sk-..."
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export PYTHONPATH="$(pwd)/src"

# 冷启动
python scripts/cold_start.py --warmup 30

# 交互式监督
python scripts/human_cli.py

# 运行完整 term
python scripts/run_term.py --simulate --steps 10

# 密封评估基线
python scripts/baseline_eval.py

# LLM 集成演示
python scripts/demo_llm.py

# 赋能维度分析
python scripts/demo_dimensions.py
```

## 核心能力

| 能力 | 状态 | 说明 |
|------|------|------|
| 自创生 Governor | ✅ | 三层决策（预演→LLM→启发式），维度生命周期管理 |
| SIWM 世界模型 | ✅ | η 失配度 + MindZero ToM + Layer 1/2/3 |
| Mnemosyne 记忆 | ✅ | 三区隔离 + 语义检索 + 跨 agent 共享 + 多巴胺门控 |
| 赋能引擎 | ✅ | 6/9 维度完整闭环（knowledge/calibration/decomposition/attention/metacognition/environment） |
| 密封评估 | ✅ | 15 任务 + 真实代码执行 + 增益三分解 |
| 铁律执行 | ✅ | G5 效用锚 + G6 熔断 + G10 密封保护 |
| 梦境管道 | ✅ | 自我模型刷新 + LLM belief 调和 + 记忆整合 |
| 成本记账 | ✅ | token/时间/算力/验证查询四账本 |
| LLM 接入 | ✅ | union-alpha via OpenRouter，代理支持，自动降级 |

## 测试

```bash
pytest tests/ -v
```

**194+ tests passing**

## 文档

- [架构白皮书](docs/PROGRESS.md)
- [搭建进度](docs/PROGRESS.md)
- [密封评估报告](docs/baseline_report.json)

## 设计原则

1. **绝不盲标 SUCCESS** — 无证据的赋能操作必须标 UNKNOWN
2. **铁律不可被改进改写** — G1-G10 是所有自创生过程的不动点
3. **三区隔离** — 自我记忆 / 经验记忆 / 代理记忆权限分离
4. **信号新鲜度 = η** — 自我模型失配度是连续生理指标
5. **维度轮换进化** — 每个 term 重点进化 2-3 个维度
6. **预演式进化** — 改进前先在世界模型中模拟后果

## License

MIT
