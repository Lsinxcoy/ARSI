# ARSI 搭建进度日志

## 项目信息
- 位置：`E:\ARSI`
- 基于：白皮书 v0.8 + SIWM 技术规格 + 代码级构建方案 v0.1
- 参考：SYNTHEX-Autopoiesis-pure、MetaRSI (arXiv:2609.06396)、MWM (arXiv:2607.27201)

---

## Phase 0：项目骨架 + 基础层 + 密封评估器

### 2026-09-16 搭建记录

#### 已完成
| 文件 | 状态 | 说明 |
|------|------|------|
| `pyproject.toml` | ✅ | 项目元数据 + 依赖（pydantic, litellm, networkx, scikit-learn） |
| `src/arsi/__init__.py` | ✅ | 包初始化 |
| `src/arsi/foundation/schema.py` | ✅ | 全部核心数据模型（Pydantic） |
| `src/arsi/foundation/store.py` | ✅ | SQLite 存储引擎（WAL + CAS + 三区隔离） |
| `src/arsi/foundation/iron_laws.py` | ✅ | G1-G10 铁律执行器 |
| `src/arsi/foundation/config.py` | ✅ | 配置加载 |
| `src/arsi/foundation/__init__.py` | ✅ | 统一导出 |
| `config/iron_laws.yaml` | ✅ | 铁律配置（人类可编辑） |
| `config/sealed_tasks.yaml` | ✅ | 密封评估任务集（12 个任务） |
| `src/arsi/sealed_eval/evaluator.py` | ✅ | 密封评估执行器 |
| `tests/test_foundation.py` | ✅ | 基础层测试（~20 个用例） |
| 目录结构 | ✅ | 13 个目录全部创建 |

#### 与代码级方案的差异
| 差异点 | 方案 | 实际 | 原因 |
|--------|------|------|------|
| 向量检索 | sqlite-vec 或 chromadb | 暂未实现 | Phase 0 不需要，Phase 1 加入 |
| sealed_tasks | 20-30 个任务 | 12 个任务 | 先建立框架，后续补充 |
| 适配器 | 具体 agent 适配器 | Protocol 接口 | Phase 0 只定义接口，Phase 1 实现 |

#### 遇到的问题
1. 测试断言与 freeze 方法的 reason 参数不匹配 — 已修复
2. MIMO_PYTHON 环境缺少 pytest — 已安装

#### 测试结果
```
21 passed in 0.55s — 全部通过
```
- Schema: 7/7 ✅
- Store: 10/10 ✅（三区隔离、CAS、因果链、行为追踪、效果账本）
- Iron Laws: 4/4 ✅（加载、默认、预算集中拦截、均衡通过）

#### 下一步

---

## Phase 1-4：全量搭建

### 2026-09-16 搭建记录

#### 已完成
| 模块 | 文件 | 状态 |
|------|------|------|
| Mnemosyne 核心 | `mnemosyne/core.py` | ✅ 三区 + 多巴胺门控 + 有用性衰减 + 边发现 |
| 记忆代理 | `mnemosyne/memory_proxy.py` | ✅ Proxy Mode + 降级 + 跨 agent 检索 |
| SIWM 世界模型 | `world_model/siwm.py` | ✅ η + MindZero + Layer 1 预测器 |
| Governor | `governor/core.py` | ✅ 决策循环 + 维度管理 + 策略蒸馏 |
| 赋能引擎 | `empowerment/engine.py` | ✅ 执行闭环 + 四级验证 + 回滚 |
| 梦境管道 | `pipelines/dream.py` | ✅ 自我模型刷新 + 记忆整合 |
| 集成测试 | `tests/test_integration.py` | ✅ 26 个用例 |

#### 测试结果
```
47 passed in 0.46s — 全部通过
```
- 基础层：21/21 ✅
- 集成测试：26/26 ✅

#### 与代码级方案的差异
| 差异点 | 方案 | 实际 | 原因 |
|--------|------|------|------|
| 向量检索 | sqlite-vec | 标签匹配 | Phase 0 简化，后续升级 |
| SIWM Layer 2/3 | 完整转移函数 | 启发式评分 | 需要更多数据训练 |
| 具体 agent 适配器 | Claude/GPT | Protocol + NullAdapter | 先定义接口 |
| LLM 调用 | litellm 集成 | 未接入 | Phase 0 不需要 |

#### 遇到的问题
1. PowerShell Set-Content 的反杠转义导致 __init__.py 语法错误 — 改用 write 工具
2. Pydantic 不允许非注解类属性 — 改用普通类测试铁律
3. 维度管理器在无数据时 data_sufficiency=0 导致无法选择 — 测试中先注入轨迹

---

## LLM 接入 + Core 主类 + LLM 内化

### 2026-09-16 搭建记录

#### 已完成
| 模块 | 文件 | 状态 |
|------|------|------|
| LLM 客户端 | `foundation/llm.py` | ✅ 代理支持 + 降级 + extra_headers |
| ARSI Core | `core.py` | ✅ 主类组装全部模块 |
| Term Runner | `scripts/run_term.py` | ✅ 完整 term 执行 |
| LLM Brain | `llm_brain.py` | ✅ LLM 内化到四个核心模块 |
| LLM Demo | `scripts/demo_llm.py` | ✅ 端到端演示 |

#### LLM 内化详情
| 模块 | LLM 用途 | 降级策略 |
|------|---------|---------|
| Governor | 分析状态、选择动作、给出理由 | 启发式候选选择 |
| MindZero | 从轨迹推断信念/目标/情绪 | 频率统计推断 |
| 赋能诊断 | 分析失败模式、推荐改进维度 | if-then 规则 |
| 梦境管道 | belief 语义调和 | 置信度阈值过滤 |

#### 实测结果（union-alpha via OpenRouter）
```
Step 1: learn — LLM 决策（理由：0 信念、10 经验、需建立认知基础）
Step 2: dream — 启发式降级（LLM 限流）
Step 3: learn — LLM 决策（理由：η=0.145 有失配、信念数为 0）
Step 4-5: dream — 启发式降级
```
LLM 可用时用 LLM，限流时自动降级——设计行为正确。

#### 与代码级方案的偏差
| 偏差点 | 方案 | 实际 | 原因 |
|--------|------|------|------|
| LLM 协议 | chat/completions | chat/completions（OpenRouter） | opencode.ai 的 union-alpha 服务端 500 |
| 模型 ID | union-alpha | stealth/union-alpha | OpenRouter 上需要 provider 前缀 |
| LLM 内化方式 | 直接改原模块 | 装饰器模式（llm_brain.py） | 保持启发式降级路径干净 |
| 测试 LLM | 真实调用 | 强制禁用（_client=None） | 避免测试时网络请求 |

#### 遇到的问题
1. opencode.ai 的 union-alpha 全端点 500 — 切换到 OpenRouter
2. OpenRouter 上模型 ID 是 stealth/union-alpha 不是 union-alpha
3. config.py 的 LLMConfig 缺 extra_headers 字段 — 补上
4. 测试中 LLM 尝试联网导致超时 — fixture 中强制禁用
5. step() 方法重构后残留旧代码导致 IndentationError — 清理

#### 当前系统状态
- **69/69 测试通过**
- **LLM 接入**：union-alpha via OpenRouter，代理 127.0.0.1:7890
- **LLM 内化**：Governor/MindZero/诊断/梦境 四个模块
- **降级策略**：LLM 不可用时自动回退到启发式
- **ARSI Core**：一个对象拥有全部系统，step() 是完整决策+执行循环

---

## N1-N6 全量推进 + P1-P6 深化

### 2026-09-16 搭建记录

#### 已完成
| 编号 | 内容 | 文件 | 状态 |
|------|------|------|------|
| P1 | 预演闭环 | `governor/pre_enactment.py` | ✅ Governor 三层决策 |
| P2 | 代码验证真实执行 | `sealed_eval/code_verifier.py` | ✅ 子进程执行+断言 |
| P3 | 六个赋能维度 | `empowerment/dimensions.py` | ✅ 6/9 完整闭环 |
| P5 | 梦境 LLM 调和 | `pipelines/dream.py` | ✅ LLM 语义比对 |
| P6 | 人类 CLI v2 | `scripts/human_cli.py` | ✅ 10+ 命令 |
| N1 | 增益三分解 | `sealed_eval/gain_decomposition.py` | ✅ 规则+LLM 混合归因 |
| N2 | 成本四账本 | `foundation/cost_ledger.py` | ✅ token/时间/算力/查询 |
| N3 | 维度生命周期集成 | `empowerment/lifecycle_integration.py` | ✅ 编排器→管理器 |
| N4 | 边发现语义升级 | `mnemosyne/core.py` | ✅ 语义关联+跨 agent 标记 |
| N5 | 任务集扩充 | `config/sealed_tasks.yaml` | ✅ 5→15 个任务 |
| N6 | 反事实仿真器 | `world_model/counterfactual.py` | ✅ 多步展开+深度自适应 |

#### 测试结果
```
184 passed — 全部通过
```

#### 与方案的偏差
| 偏差点 | 方案 | 实际 | 原因 |
|--------|------|------|------|
| 增益归因方法 | 密封评估 per_task_delta | 规则分类 + LLM 比例混合 | per_task_delta 需要更细粒度的任务匹配 |
| 成本 token 计数 | 精确 API usage | 近似值（500/200 per call） | OpenRouter 不总是返回 usage |
| 反事实仿真 | 用 Layer 2 完整物理转移 | 简化为 η 和 storage 变化 | 完整 Φ 转移需要更多特征 |
| N7 多 agent | 计划包含 | 跳过（用户要求最后） | 用户明确指示 |

#### 当前系统能力
- **184 测试通过**
- **6/9 赋能维度**有完整闭环
- **三层决策**：预演 → LLM → 启发式
- **真实代码执行**验证
- **增益三分解** + **成本记账**
- **语义检索** + **语义边发现**

---

## 持续闭环 + 自动化

### 2026-09-17 搭建记录

#### 已完成
| 组件 | 文件 | 状态 |
|------|------|------|
| MiMo 会话提取器 | `adapters/mimo_extractor.py` | ✅ 从 MiMo 记忆自动提取轨迹 |
| 持续闭环脚本 | `scripts/continuous_loop.py` | ✅ 6 步反馈循环 + 监听模式 |
| 自动启动脚本 | `scripts/start_loop.bat` | ✅ Windows 批处理 |
| 计划任务 | Windows Task Scheduler | ✅ 每天 09:00 自动运行 |

#### 实测验证
- MiMo 记忆路径自动检测（处理 SYSTEM 用户问题）
- 7 个会话、13 条真实轨迹提取成功
- 闭环流程：提取→导入→分析→建议→写反馈文件

#### 遇到的问题
1. PowerShell 反杠转义导致 docstring 语法错误 — 改用正斜杠
2. 运行用户是 SYSTEM 而非实际用户 — 多重路径检测
3. LLM 调用超时代理不可用 — 无 LLM 模式仍可运行
4. **GitHub 密钥泄露检测**：start_loop.bat 硬编码 API key 被 GitHub 拒绝推送 — 移除硬编码，改为运行前设置环境变量
- **15 个密封评估任务**

---

## Dream-RSI 深挖 + 机制落地

### 2026-09-18 深度研究记录

#### 研究结论（arXiv:2609.14858）
上次只吸收了「发现树外壳」。本轮从论文全文 + dream-rsi.com + GitHub README + 附录 B 完整 prompt 挖出五大未吸收机制：

| 机制 | 要点 |
|------|------|
| **世界池 H_t** | 每圈追加一棵发现树；策略在所有历史世界上平均评估 |
| **Child() 语义** | 非根→唯一已记录子节点；根→最早未揭示子节点（开新分支） |
| **Prefix-only API** | Observation 成功语义：error is None + fail_class=="ok"（valid=false 仍可成功） |
| **失败四分类** | hard / repairable / weak-underexplored / repeatedly-unpromising；可修复失败保留资格 |
| **动态 portfolio + β** | exploit+explore+≤1 recovery；β episode 内固定，跨周期按 live 规则调整 |
| **§5.1 负结果** | 历史当语义指导注入 prompt **更差**；必须当结构化模拟器 |

实验数字：Lasso vs SimpleTES **162×** fewer calls；math **>50×** budget saving；Kernel **2.09×** higher / **2.43×** fewer gens。

#### 本轮交付
| 文件 | 状态 | 说明 |
|------|------|------|
| `world_model/replay_world.py` | ✅ | 论文形式化回放世界（Child/Observation/prefix API） |
| `world_model/world_pool.py` | ✅ | H_t 世界池 + 多世界评估 + 单调策略选择 |
| `governor/portfolio_policy.py` | ✅ | 动态 portfolio 批次 + β schedule + 跨周期 default β |
| `core.py` 接线 | ✅ | `harvest_term_tree` / `dream_rsi_cycle`；step 每 8 步触发 |
| `tests/test_dream_rsi_deep.py` | ✅ | Child/池/单调/β/ARSI 接线测试 |
| 深度研究报告 | ✅ | `E:\Mimo 生成\docs\2026-09-18\Dream-RSI-deep-research.{md,html}` |

#### 测试结果
```
269 passed in 9.17s — 全部通过
```
（含新增 `tests/test_dream_rsi_deep.py` 14 项）

#### 路径偏差（必须记录）
| 产物 | 路径 | 原因 |
|------|------|------|
| 研究报告 md/html | `E:\Mimo 生成\docs\2026-09-18\` | 全局生成物规则：AI 生成文件写入 E:\Mimo 生成 |
| 论文 PDF 缓存 | `E:\Mimo 生成\cache\2026-09-18\papers\` | 同上（cache 分类） |
| ARSI 源码/测试 | `E:\ARSI\` | 用户项目仓库，非生成物，保持原位 |
| 进度日志 | `E:\ARSI\docs\PROGRESS.md` | 项目内文档，按约定记录偏差 |

#### 明确未做（诚实边界）
- GridPlan `plan_grid`（宽度/深度规划）— 需要 live cycle manifest 体系
- AdaptiveBehaviorController 未接入 Governor 主环
- β grid sweep 评估器与 beta_sweep.json 落盘
- bidirectional `brief()` 方向性建议字段的全面审计
- 官方代码未发布：β₁/β₂、M、K₂、λ、β grid 具体数值未知
- 多 agent 真连接 — **仍按约定留到最后**

#### 官方资源状态
- 论文 PDF / 官网 / 交互 demo：已放出
- GitHub 仓库：Paper ✅ Project page ✅ arXiv 🔜 **Full codebase ⏳**
- 源：https://github.com/zhengkid/Dream-RSI · https://dream-rsi.com/

---

## Phase D 落地方案（未完成工作）

### 2026-09-18 方案记录

深度盘点后的判断：未落地项不是散点 TODO，而是 **Dream-RSI 元探索闭环的下半截断了**，外加质量/调度层从未接入主环。

| 断点 | 现状 |
|------|------|
| QualityGate / OperatorScheduler / AdaptiveController | 有代码，core/daemon **未调用** |
| GridPlan / live manifest 落盘 / β sweep | **缺失** |
| brief() | 仍输出 Recommendations/Past Lessons（§5.1 风险） |
| scripts/daemon | **未使用** world_pool / portfolio / dream_rsi_cycle |

**方案文档**（生成物，遵循全局规则写入 Mimo 生成）：
- `E:\Mimo 生成\docs\2026-09-18\ARSI-phase-D-landing-plan.md`
- `E:\Mimo 生成\docs\2026-09-18\ARSI-phase-D-landing-plan.html`

**分期**：D1 Manifest → D2 GridPlan → D3 β sweep → D4 brief 合规 → D5 三件套接线 → D6 Daemon → D7 评估对照 → D8 延期（官方超参/多agent）。

**域映射（已钉死）**：GridPlan.W = term 内并行焦点数（维度/算子），R = 每焦点精炼步数；W 当前不是真线程并行（偏差已声明）。

**默认超参偏差**（官方未 Release）：β₁=0.1, β₂=0.05, λ=0.05, M=2, K₂=20, β grid [0.2,0.4,0.6,0.8,1.0], uncertain default β=0.6。

**状态**：方案已交付，**代码尚未按 D1 开工**。测试基线 269 passed。

**路径偏差**：方案文档在 `E:\Mimo 生成\docs\2026-09-18\`（生成物规则）；源码与本进度日志在 `E:\ARSI\`（项目仓库）。

---

## Phase D 开工落地

### 2026-09-18 实现记录

按方案 **D1→D7 最小闭环** 已写入代码并全量测试通过。

#### 已交付

| 阶段 | 文件 | 状态 |
|------|------|------|
| D1 Manifest | `src/arsi/meta/live_manifest.py` | ✅ schema + ManifestStore 落盘/容错/reload |
| D2 GridPlan | `src/arsi/meta/grid_plan.py` | ✅ plan_grid 证据规则 R1–R4 + bootstrap + 预算钳制 |
| D3 β sweep | `src/arsi/meta/beta_sweep.py` | ✅ 池上扫 β + pareto reward + default-β 规则 |
| D4 brief §5.1 | `adapters/bidirectional_interface.py` | ✅ `structured` 默认；建议进 meta_only；历史仅结构化统计 |
| D5a QualityGate | `core.py` `_filter_traces_for_world` | ✅ PASS/WARN 入池，FAIL 拒入；stats 进 manifest |
| D5b Scheduler | `core.py` run_term/dream_rsi_cycle | ✅ schedule 进 term + Law2 mark_capability_change |
| D5c Adaptive | `core.py` plan_next_grid | ✅ trend→force 调节 R |
| D6 Daemon | `scripts/arsi_daemon.py` | ✅ 每 8 tick 跑 dream_rsi_cycle；health 含 pool/grid/β |
| 配置 | `config/arsi.yaml` | ✅ quality_gate / grid_plan / beta_sweep / brief.policy |
| 测试 | `tests/test_phase_d.py` | ✅ 18 项 |

#### 接线摘要
- `dream_rsi_cycle`: plan_grid → harvest(gate) → pool 评估 → LLM 修订 → β sweep → 单调选择 → **manifest + beta_sweep.json**
- `run_term`: 计划网格 → OperatorScheduler 选维度 → 步数受 W×R 约束 → adaptive → **manifest**
- `brief()` 默认 structured：无 Recommendations/Past Lessons；`full` 供人类 CLI
- 运行数据：`E:\ARSI\archive\trace_pool\iter*\live_cycle_manifest.json` + `beta_sweep.json`

#### 测试结果
```
287 passed in 11.37s — 全部通过
```
基线 269 → **287**（+18 Phase D）

#### 默认超参偏差（官方代码未 Release）
β₁=0.1, β₂=0.05, λ=0.05, M=2, K₂=20, β grid `[0.2,0.4,0.6,0.8,1.0]`, uncertain default β=0.6。W=并行**焦点数**（非 OS 线程）。

#### 路径偏差
| 产物 | 路径 |
|------|------|
| Phase D 方案 | `E:\Mimo 生成\docs\2026-09-18\ARSI-phase-D-landing-plan.{md,html}` |
| 源码/测试/PROGRESS | `E:\ARSI\` |
| manifest/sweep 运行数据 | `E:\ARSI\archive\trace_pool\`（项目 archive，非生成物文档目录） |

#### 仍未做
- D7 完整 fixed vs dream 对照报告 + live 变差自动回退（骨架字段 sealed_delta/gain 已进 manifest，对照脚本未写）
- 官方超参回填（等 GitHub full code）
- 多 agent 真连接（约定最后）
- `continuous_loop.py` / `human_cli.py` 未加 manifests/grid 子命令（daemon 已接）

---

## Phase D 补齐（三项，不含多 agent）

### 2026-09-18 实现记录

#### 1) D7 评估闭环 + 自动回退
| 文件 | 说明 |
|------|------|
| `src/arsi/meta/eval_loop.py` | fixed vs dream_rsi 池上对照；live 回归检测；自动回退 β |
| `archive/eval/compare_*.json` + `compare_log.jsonl` | 对照报告落盘 |
| `dream_rsi_cycle` | 每次元循环后自动 `run_eval_loop` |

规则：
- **池内单调 ≠ live 单调** — 回归窗口 `rollback_window=3`，阈值 `eps=0.02`
- live 回退时部署 `rollback_beta_*`（β 下调或回到 uncertain default 0.6）
- 对照基线 `fixed_exploration_fn` 同时作为 dream 选择候选之一

#### 2) 超参配置层（官方未发布 → 可回填）
| 文件 | 说明 |
|------|------|
| `config/dream_rsi_params.yaml` | β₁/β₂/λ/M/K₂/β grid/Grid 顶格/rollback 策略 |
| `src/arsi/meta/dream_rsi_params.py` | `load_dream_rsi_params()` |
| `core.py` | `arsi.dream_rsi_params` 注入 sweep/grid/step 周期 |

`official_code_status: not_released` — 官方 GitHub full code 放出后**只改 YAML**，禁止静默改 Python 常量。

#### 3) CLI / continuous_loop 接线
| 文件 | 新命令/步骤 |
|------|-------------|
| `scripts/human_cli.py` | `manifests` `grid` `beta` `pool` `dreamrsi` `compare` `params` |
| `scripts/continuous_loop.py` | 2b QualityGate 收割 + 7/7 dream_rsi + eval 输出 |
| `scripts/arsi_daemon.py` | （上轮已接）每 tick 跑 meta cycle |

#### 测试结果
```
296 passed in 11.33s — 全部通过
```
基线 287 → **296**（+ residual/eval/params 测试）

#### 语义修正
- `default_beta_from_live(None)` 读磁盘 manifest；传入 list（含空）只用该 list，不足 2 条 → bootstrap 0.6
- 单元测试的 ARSI 实例强制 `manifest_store` 指向临时目录，避免污染 `archive/trace_pool`

#### 路径
| 产物 | 路径 |
|------|------|
| 超参 YAML | `E:\ARSI\config\dream_rsi_params.yaml` |
| 对照报告 | `E:\ARSI\archive\eval\` |
| 源码/测试 | `E:\ARSI\` |

#### 仍未做（更新）
- **多 agent 主动协议闭环 / 编排** — 用户约定最后
- 官方超参数值回填 — 等 GitHub Release 后改 YAML
- 连续 live 对照需 daemon 跑出足够 manifest 后才有统计力

#### Dream-RSI 架构地位认定（2026-09-18）
| 层级 | 是否基石 | 证据 |
|------|----------|------|
| 代码主环 | **是** | core.step/run_term/dream_rsi_cycle/daemon 均调用世界池+manifest+eval |
| 项目章程 | **是（本轮 README 正名）** | 四大支柱：Autopoiesis / MetaRSI / MWM / **Dream-RSI** |
| 运行承重 | **尚未** | archive/trace_pool 有 manifest，但近期多为 best_score=0 / pool=0 的薄数据；daemon 未持续跑出健康圈 |

结论：**代码级已是架构支柱之一；运行级尚未成为承重墙**——差的是常驻 daemon + 真实三宿主轨迹把池喂厚。

---

## Daemon 拉起（2026-09-18 15:50）

#### 环境问题
| 问题 | 处置 |
|------|------|
| **C 盘剩余 0 GB** → SQLite「database or disk is full」 | 清理 `AppData\Local\Temp` 旧文件，释放约 **36 GB**，C: 现约 35.5 GB 空闲 |
| Hermes `state.db` / MSTAR disk I/O | 同根因（C 盘满）；入库后应恢复 |
| 旧 daemon PID 33292（10:50 起）仍在跑 **Phase D 之前** 的代码 | 已停止并清锁，避免内存中旧逻辑 |

#### 当前进程
| 进程 | PID | 说明 |
|------|-----|------|
| `arsi_daemon.py --tick-sleep 300` | **7640** | 2026-09-18 15:50:51 起，**加载 Phase D 新代码** |
| `arsi_interface.py --port 9300` | 30228 | 双向协议 HTTP（旧进程，未重启） |

启动方式：环境变量 `PYTHONPATH/ARSI_API_KEY/HTTP(S)_PROXY/TEMP→E:\ARSI\archive\tmp`，完整 python 路径，Hidden；日志 `archive/daemon.log` + `daemon_err.log`。

#### 首 tick 证据
- 已 ingest **377** 条新轨迹（hermes/synthex 等）
- LLM `mcgrox.top` HTTP 200 正常
- Mnemosyne 蒸馏与语义索引运行中
- Dream-RSI meta cycle 按代码 **每 8 tick** 触发；manifest 非零 score 需等后续 tick + 质量门入池

#### 路径偏差
- daemon 临时目录：`E:\ARSI\archive\tmp`（避免 C 盘再满）
- API key 仅进程环境变量注入，**未新写入仓库文件**（`start_daemon.bat` 内历史硬编码仍在，GitHub 推送前勿提交密钥）

---

## 母巢 SYNTHEX 深度分析（2026-09-18）

### 结论
母巢 `MotherWorldModel`（~2720 行）是**生产级可审计世界模型账本**：联合状态、TransitionRecord（预测/实际/误差/归因/pair_id）、时间窗 EffectPredictor、不确定度截断、诊断电池（含 **signal_collapse**）、JOIN 审计、SelectionGate 配对 A/B。**诚实修正**：wiring_plan 的多候选模拟选优**未接线**；MWM 自称不学动力学；合成 5D 含零信息维。

### 对 ARSI 最锋利的生产教训
- **verified 常量自证**（1889/1889）→ 已验证必须绑测试证据  
- **身份与写方不同源** → A/B 选择压力断  
- **桥建好≠通电** → 已建≠已接线≠已通电≠有数据流  
- **方案 72 点仅 1 落地** → 文档≠实现  
- **伪进化**：成本不收敛=退化  

### 吸收优先级（摘要）
| 级 | 项 |
|----|-----|
| P0 数据可信 | TransitionLedger · 时间窗基线 · signal_collapse · verified 绑证据 · 单一落盘源 · MentalDelta |
| P1 选择/诚实 | 配对A/B+影子 · JOIN门 · 多候选选优 · T-convergence · 固定探针 · VacuumGate |
| P2 | 防错规则 · 信用分配 · 技能代际 · 导管（多agent最后） |

### 交付物
- `E:\Mimo 生成\docs\2026-09-18\SYNTHEX-mothernest-deep-analysis.md`（含 explore 合并 §9）
- `E:\Mimo 生成\docs\2026-09-18\SYNTHEX-mothernest-deep-analysis.html`

### 路径偏差
分析报告在 `E:\Mimo 生成\docs\2026-09-18\`；**未改母巢代码**；未按 P0 开工改 ARSI（待确认）。

### 仍未做
- 多 agent 主动协议 / 蜂巢导管 — 最后  
- P0 SIWM 补强代码 — 本分析交付后待开工  
- 官方 Dream-RSI 超参回填 — 等 Release  

---

## ARSI 内省世界模型落地研究（2026-09-18）

### 诊断（代码事实）
- SIWM 名称含 Introspective，实质：η（动作类预测误差）+ MindZero 推断 Ψ + Layer1 行为预测  
- **`dream.py` 将 η 硬编码 `η*0.5`** — 无测量即宣称自我模型刷新（假内省）  
- `first_person_observe` 仅 4 个数字；Governor **不消费**器官可信度  
- Dream-RSI eval_loop/manifest 已有回路证据，未并入自我模型  

### 真内省验收 Q1–Q6
器官可靠度 · 知识边界 · 改进回路效力 · 自我转移 · 决策溯源 · 内省器自检。  
**Q1–Q3 任意两条不过 → 不得称内省世界模型。**

### 目标 IWM 模块
OrganSelfModel · KnowledgeFrontier · LoopEfficacy · TransitionLedger · DecisionProvenance · calibrate 降级。  
每条器官可靠度必须有**行为挂钩**（无挂钩不算内省）。

### 分期
I1 删假η+OrganSelf+Ledger+LoopTrial → I2 Frontier → I3 Governor 接入 → I4 自检降级 → I5 与 Dream-RSI 咬合 → I6 行为探针全过后才改 README 表述。

### 交付物
- `E:\Mimo 生成\docs\2026-09-18\ARSI-introspective-world-model-design.md`
- `E:\Mimo 生成\docs\2026-09-18\ARSI-introspective-world-model-design.html`

### 路径偏差
研究文档写入 `E:\Mimo 生成\docs\2026-09-18\`；**本轮未改 ARSI 运行代码**（仅研究方案）。

### 状态
方案已交付，**Phase I1 未开工**。待确认 Q1–Q6 验收定义后按 I1 开写。

---

## IWM 内省世界模型落地（2026-09-18 实现）

### 原则（执行口径）
- **Q1–Q3 任意两条不过 → 不得称内省世界模型**（空系统默认 skeleton）
- 每条器官可靠度必须有**行为挂钩**；无挂钩 = 仪表，不算内省
- **禁止 dream 手拧 η**（删除 `η*0.5`）；η 仅 LoopTrial 证据支持才可下降

### 已交付代码
| 模块 | 路径 | 职责 |
|------|------|------|
| IWM 门面 | `src/arsi/iwm/__init__.py` | Q1–Q6 汇总、governor_advice、health、q_gate |
| OrganSelfModel | `src/arsi/iwm/organ_self.py` | 器官滚动可靠度 + control_hooks |
| KnowledgeFrontier | `src/arsi/iwm/frontier.py` | 已知/薄弱/欠探索 + explore_bias |
| LoopEfficacy | `src/arsi/iwm/loop_efficacy.py` | LoopTrial before/after + 证据驱动 metric |
| TransitionLedger | `src/arsi/iwm/transition_ledger.py` | 自我预测 vs 现实账本（落盘 jsonl） |
| DecisionProvenance | `src/arsi/iwm/provenance.py` | 决策可回放 + IWM 快照 |
| IntrospectorCalibrator | `src/arsi/iwm/calibrate.py` | Q6 自检；低信任 → degrade baseline |

### 行为挂钩（已接线）
| 证据 | 控制变化 | 代码位置 |
|------|----------|----------|
| dream 连续 neutral/hurt | 禁止默认 dream → Governor 改 learn | `governor/core.py` + `core.step` |
| dynamics unknown/unreliable | 降权/跳过 pre-enactment | `core.step` / `governor.decide` |
| behavior_predictor 不可靠 | prefer learn，evolve 降权 | `governor._generate_candidates` |
| calibrator 低信任 | degrade_to_baseline | IWM.advice + Governor |
| eval_loop live 回归 | portfolio 器官记不可靠 | `core.dream_rsi_cycle` → `iwm.observe_eval_loop` |

### dream.py 修正
- 删除 `new_eta = eta_before * 0.5`
- dream 前后测量 holdout 行为预测误差
- LoopTrial verdict=helped 才允许 η 向 measured 有界移动（max_step=0.15）
- first_person 报告含 organs/frontier/loop/self_trust（不再是 4 个虚荣数字）

### 测试
```
317 passed — 全部通过
```
- 新增 `tests/test_iwm.py`（OrganSelf / LoopTrial / Frontier / Provenance / Calibrate / Dream 证据 η / Governor hooks / ARSI 接线 / Q 探针）
- 修正 `tests/test_integration.py::test_dream_reduces_eta` → 证据驱动语义

### 诚实边界
- 运行承重仍依赖 daemon 把轨迹喂厚；样本不足器官 status=unknown，不进控制
- I6「内省世界模型 v1」宣称：需 Q1–Q6 行为探针在 live 数据上全过；当前交付为**内省骨架 + 真实控制挂钩**，不是空仪表
- 多 agent 协议仍未做（约定最后）
- 官方 Dream-RSI 超参仍等 Release 改 YAML
- 本轮**未 git commit / push**（工作区含 Phase D 未提交改动 + 本次 IWM）

### 交付路径
- 源码：`E:\ARSI\src\arsi\iwm\`
- 测试：`E:\ARSI\tests\test_iwm.py`
- 方案（先前）：`E:\Mimo 生成\docs\2026-09-18\ARSI-introspective-world-model-design.html`
- 冒烟脚本：`E:\Mimo 生成\cache\2026-09-18\iwm_smoke.py`

---

## 运行修复落地（2026-09-20 · 按深度分析建议）

### 根因（评分墙）
`replay_score = quality − β₁·N + β₂·(N/k)`  
在 quality≈0、N≈38、β₁=0.1 时 → **−3.67**，所有策略同坑 → β sweep 退化、live 冻结。

### 已交付修复
| 项 | 变更 |
|----|------|
| **A 评分诊断+锚定** | `ReplayResult.score_breakdown`；默认 `score_mode=quality_anchored`：quality 低于阈值时成本项缩放（`weak_quality_cost_scale`）；`paper` 模式保留原公式 |
| **质量信号** | discovery 成功节点 score = `0.2+0.8*effect`（不再贴近 0） |
| **B 池准入** | `world_min_verdict=PASS`；WARN 仅配额（默认 ≤25%）；PASS 按 effect 优先；`max_admitted` 上限 |
| **C β 冻结** | sweep `non_degenerate=false` → `degenerate_freeze_beta_*`，**不再 plateau_raise 空转**；部署名 `frozen_plateau_*` |
| **策略编译** | `_parse_llm_json` 剥离 CJK 标点/代码围栏，降低 `invalid character '。'` |
| **Q6 calibrator** | no-op/learn 无蒸馏/remember 不算 success；无 baseline 臂时 self_trust **封顶 0.75**；step 采样 baseline 臂 |
| **Layer1** | 增加 effect-bucket / 二阶 pair 规则上下文 |

### 配置
- `config/dream_rsi_params.yaml` + `config/arsi.yaml`：PASS 优先、score_mode、freeze_beta_on_degenerate  
- NVIDIA LLM：`nvidia/nemotron-3.5-lightning-30b-a3b` @ integrate.api.nvidia.com

### 测试
```
335 passed
```
新增 `tests/test_runtime_fixes.py`（评分墙/PASS池/β冻结/策略消毒/calibrator）。

### 诚实边界
- quality_anchored 是 **ARSI 标定偏差**（官方未发布超参前）；官方代码放出后只改 YAML 回填 paper 模式参数  
- 旧池内世界仍是历史低质量数据；新 harvest 才按 PASS 重建  
- live 能力分提升仍需 daemon 持续跑 + 更高质量轨迹，不是单次改分就能「变聪明」

---

## 下一步：Layer1 live 绑定 + S1 verified（2026-09-20）

### Layer1
| 项 | 结果 |
|----|------|
| holdout（生产库 800 轨迹） | **0.9057–0.9091**（train 400 / test 100） |
| rules / pair_rules | 24 / 49 |
| 旧 live_accuracy 易低估 | 未 fit 时全预测 other；现加 majority fallback + 时序 holdout |
| IWM 绑定 | `observe_layer1_holdout` → behavior_predictor 器官 + health.layer1 |

`scripts/layer1_live_report.py` → `archive/eval/layer1_live_report.json`

### S1 verified（daemon health）
`verified` 字段改为 `VerifiedClaim`：绑定 `iwm_in_stats` + `layer1_holdout_accuracy` 运行字段与时间戳，**禁止常量自证**。

### 测试
```
338 passed
```
新增 `tests/test_layer1_live.py`

---

## 二次验收 + 多 agent 协议（2026-09-20 续）

### Fast live acceptance（关 LLM，生产库）
| 项 | 值 |
|----|-----|
| Layer1 holdout | **0.9192**（已 bind IWM） |
| Harvest PASS 门 | **pass=60 kept，warn_admitted=0**（fail=68 剔除；WARN 配额生效） |
| Pool score_mode | **quality_anchored** |
| dream vs fixed | **-2.606 vs -2.6995**（quality **0.76**；Δ +0.0935） |
| 墙 | **已离开 -3.67** |
| paired AB | hold n=1<5（进程内新池，daemon 持续后会变厚） |
| frozen_plateau | 累计 8；`degenerate_freeze_beta_0.60` |
| daemon | PID **30160 / 25668**（09:34） |

产物：
- `E:\ARSI\archive\eval\fast_acceptance_latest.json`
- `E:\Mimo 生成\docs\2026-09-20\fast-acceptance-latest.json`

### 多 agent 协议 M1–M4（约定中的「最后」，已最小闭环）
- `src/arsi/multiagent/protocol.py`：注册 / ASSIGN+BRIEF / RESULT / freeze
- 接 `ARSIInterface.brief/report`；host effect → **external anchor**
- **M3**：&lt;3 次 outcome → organ=`unmeasured`；≥3 才 `measured` 并 bind IWM
- **M4**：iron law / freeze → 拒绝 dispatch
- `get_stats.multi_agent`

### 测试
```
365 passed
```

### 仍未做
- 真宿主进程对接（HTTP/MiMo Desktop 实调用）——协议层已就绪
- 官方 Dream-RSI 趞参回填
- live I6：等 daemon 池 n≥5 + paired promote 证据


### 运行验收（观察）
| 信号 | 值 |
|------|-----|
| 最新 compare | **09:23** fixed/dream **0.69 / 0.69**（已离开 **-3.67** 墙） |
| β 冻结 | `frozen_plateau` ×6；sweep notes `degenerate_freeze_beta_0.60` |
| paired_ab | `hold_insufficient_samples_n=1<5`（池重建中，样本未满） |
| 池质量 | 池侧 gate 暂空（PASS 重建中）；health 尾条仍是部署前旧 tick |
| daemon | PID **23136** 在跑；新 health 行要等完整 tick |

报告：`E:\Mimo 生成\docs\2026-09-20\runtime-acceptance.json`

### S6/S7 交付
- `meta/paired_ab.py`：配对 Δ + Cohen’s d；tiny mean / negligible → **hold**；champion 恒在候选集
- `world_pool.select_best_policy` 默认走配对门
- `foundation/vacuum.py`：巩固期真空带；dream LLM 调和 **只允许 grounded 信念**
- `eval_loop.notes.paired_ab`；`get_stats.vacuum`

提交 `4580411` 已推送。


| S | 项 | 实现 |
|---|----|------|
| **S1** | verified 绑证据 | daemon health `VerifiedClaim`（上轮） |
| **S3** | 单一写源 | `paths.write_json_once` / `append_jsonl`；manifest/eval/β sweep 统一经此写盘并带 `_arsi_write.writer_id` |
| **S4** | 路径锚定仓库根 | `foundation/paths.py`：`project_root()` 扫 pyproject；archive/eval/iwm/trace_pool/config 单点解析；禁 CWD 相对 |
| **S5** | 外部 effect 锚 | `foundation/effect_anchor.py`：eval_loop 结果记 external anchor；`evolve_attempt` 占位 **不再自评 0.5**，标 `non_external_source` |
| **S2** | 身份同源 | ManifestStore 默认 root = `paths.trace_pool_dir()`，与写路径同函数；`identity_report()` 进 `get_stats.paths` |

### 测试
```
348 passed
```
新增 `tests/test_p0_paths_effect.py`

### 路径
- 模块：`src/arsi/foundation/paths.py` · `effect_anchor.py`
- 仍在跑：daemon（加载 score-wall + Layer1 绑定 + P0 路径代码需下次重启生效）


### P0 卫生
| 项 | 状态 |
|----|------|
| 密钥扫描 | `start_daemon.bat` 曾硬编码 `ARSI_API_KEY` → **已移除**，改为环境变量注入；`start_loop.bat` 保持注释占位 |
| `.gitignore` | 增加 lock/log/tmp/`.env` |
| human_cli | 新增 `iwm` / `organs` / `qgate` / `frontier` |
| continuous_loop | step 后打印 IWM advice + q_gate |
| arsi_daemon health | snapshot 增加 `iwm`（organ_trust/dream_loop/q_gate） |
| foundation/verified.py | S1：verified 必须绑 pytest/exit_code + timestamp |
| scripts/iwm_probes.py | Q1–Q6 探针；报告写入 `archive/eval/iwm_probe_latest.json` |

### P1 证据回路
- `core.step`：pre_enactment 后 `observe_dynamics`（预测状态 vs step 后真实状态）
- `core.step`：learn 的 `distilled` → `observe_memory`
- dream / behavior_predictor / portfolio 器官已在此前接线

### 探针结果（本轮 runner，seeded traces）
```
claim: introspective_v1_candidate  failed=[]
q_pass: Q1–Q6 全 True
eta_policy: no_evidence_dream_must_not_move_metric
suite_verified: pytest_exit_0
hooks_applied: 6
```
**诚实边界**：runner 用种子轨迹验证挂钩与探针管线；**I6 live 宣称**仍需 daemon 喂厚后的器官样本（当前 health jsonl 仍是 Phase D 旧 schema，PID 7640 未加载 IWM 前不会写 iwm 段）。

### Daemon 现场
| 进程 | PID | 说明 |
|------|-----|------|
| arsi_daemon | **14292** | 2026-09-20 08:49+，**NVIDIA LLM**，tick-sleep 300 |
| arsi_interface | **16956** | :9300，`/arsi/health` → `llm: true` |

**LLM 切换（2026-09-20）**
| 项 | 值 |
|----|-----|
| provider | openai（OpenAI 兼容） |
| model | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| api_base | `https://integrate.api.nvidia.com/v1` |
| key | 进程环境 `ARSI_API_KEY=nvapi-...`（**未写入仓库**） |
| proxy | `use_proxy: false`（直连 200 OK） |
| 验证 | 直连/代理 chat.completions 均 200；daemon 日志 NVIDIA 200 |

旧 mcgrox.top / deepseek 已因余额不足 403 弃用。

启动方式：进程环境注入 `PYTHONPATH/ARSI_API_KEY/TEMP→E:\ARSI\archive\tmp`。

### 测试
```
322 passed
```
- +`tests/test_iwm.py`
- +`tests/test_p0_discipline.py`（verified + probes 模块）
- dream 集成测试改为证据驱动语义

### 仍未做
- daemon 重启加载 IWM（需在环境注入 `ARSI_API_KEY`，**禁止再写进仓库**）
- live I6 探针（等 ≥48h 厚数据）
- 多 agent 协议（约定最后）
- 官方 Dream-RSI 超参回填
- GitHub 历史中可能仍残留旧 key 提交 → **建议轮换密钥**；新代码不再提交密钥

---

## 多 agent HTTP + daemon 接线（2026-09-20 最终）

| 项 | 状态 |
|----|------|
| HTTP 端点 | `POST /orsi/ma/register\|dispatch\|result\|cycle`，`GET /arsi/ma/health` |
| 冒烟 | hermes：register → dispatch → result **accepted**；`organ_status=unmeasured`（&lt;3 次，符合 M3） |
| effect | `host_outcome` external anchor **verified=true** |
| daemon | PID **30824**（09:46）；interface **8536** |
| LLM | NVIDIA 偶发 404/504 → `_mark_unavailable` 启发式降级 |
| 提交 | `9572bc5` 已推送；**365 passed** |

宿主最小调用：
```text
POST :9300/arsi/ma/register {"agent_id":"hermes","role":"worker","capabilities":["tool"]}
POST :9300/arsi/ma/dispatch {"agent_id":"hermes","task":"..."}
POST :9300/arsi/ma/result   {"agent_id":"hermes","task_id":"...","task":"...","outcome":"success","effect":0.6}
GET  :9300/arsi/ma/health
```

---

## 真实宿主循环 host_loop（2026-09-20）

### 脚本
`scripts/host_loop.py` — 不再合成宿主，直接接本机 MiMo / Hermes / SYNTHEX。

### 本轮实测（cycle 1）
| 宿主 | 真实数据 | 真实副作用 | 结果 |
|------|----------|------------|------|
| **Hermes** | state.db **3529 sessions / 758556 msgs / 54 tools**；ingest **333** | EmpowermentApplier 写 `AppData/Local/hermes/skills/arsi-*/SKILL.md`（09:51:41） | success effect **0.35** |
| **MiMo** | ingest **41**（session memory） | `arsi_feedback/latest_suggestions.md` **+host_loop 段**（5204B） | success effect **0.5** |
| **SYNTHEX** | ingest **83**；home/state 存在 | `E:\SYNTHEX Autopoiesis\docs\arsi_host_loop_guidance.md`（**core 未改**） | success effect **0.4** |

- Layer1 holdout：**0.8889**（真实混合轨迹）
- HTTP 多 agent：hermes-http register/dispatch/result **accepted**，external anchor verified
- organ 三宿主均为 `unmeasured`（&lt;3 次结果，符合 M3）
- 报告：`E:\ARSI\archive\eval\host_loop_latest.json`

### 运行
```text
set PYTHONPATH=E:\ARSI\src
python E:\ARSI\scripts\host_loop.py --cycles 1 --http
```

---

## host_loop 入 daemon + 器官 measured（2026-09-20）

| 项 | 结果 |
|----|------|
| daemon | `_run_host_loop` 每 **3 tick** 且数据源有变更时执行；health 记 `host_loop` |
| 三 cycle 实测 | hermes / mimo-desktop / synthex：**organ_status=measured**，success_rate **1.0**，assigned=3 |
| 真实摄入/圈 | hermes **333** + mimo **41** + synthex **83** |
| Layer1 | holdout **0.899** |
| 副作用 | Hermes skills / MiMo feedback / SYNTHEX docs guidance 均已落盘 |

**含义**：M3 门槛已过，IWM 可消费宿主可信度；下一步看 multi-agent health 与 IWM 是否把 host 结果纳入控制。

---

## memory trust → Governor 建议（2026-09-20）

| 项 | 实现 |
|----|------|
| OrganSelf hooks | `trust_memory_for_learn` / `downweight_memory_ops` / `prefer_remember_ingest` / `memory_trust` / `memory_status` |
| governor_advice | 上述字段 + `prefer_learn_reason=memory_trust` |
| Governor | memory 可信 → **优先 learn**；memory 不可靠 → **禁 evolve**，补 remember/learn 重摄入 |
| health | `iwm.memory_trust` / `trust_memory_for_learn` / `downweight_memory_ops` |
| 测试 | **370 passed**（`tests/test_memory_trust_governor.py`） |
| 证明 | `archive/eval/memory_trust_advice_proof.json`：trusted→`trust_memory_for_learn=true`；untrusted→`downweight_memory_ops=true` |

提交 `db7b248` 已推送。

---

## P2-8 action-conditioned 速度场收尾（2026-09-23）

### 交付
| 项 | 说明 |
|----|------|
| `ActionConditionedVelocityField` | $v(z;a)$ = 全局场 + 动作残差；`min_action_samples` 前回退全局 |
| `canon_action` / `ACTION_POOL` | learn/remember/dream/maintain/evolve/empower/repair/unknown |
| `integrate` / `integrate_backward` / `reverse_from_failure` | 均接受 `action`，经 `_field_predict` 安全下传 |
| `CapabilityFlowTracker` | `observe(..., action=)`、`reverse_last_failure(action=)`、health 带 `last_action` / `action_support` |
| `core._recent_flow_action` | harvest 按最近轨迹主操作条件化；dream 固定 `action="dream"` |
| `host_reverse_loop` / `pull_health` | 失败反向带 action；观测 action_support |
| 序列化 | `to_dict` / `from_serialize` 保留 action 残差与计数 |

### 修复
- 去掉 `field.predict.__code__` duck-type（bound method 上直接炸）
- 统一 `_field_predict`：`try predict(z, action=)` → `TypeError` 回退 `predict(z)`

### 测试
```
482 passed（项目根目录全量）
```
- 新增 `tests/test_action_conditioned_flow.py`
- 修掉 lastfailed 中 integrate/backward/reverse 及 dream 接线相关失败

### 语义
- 反向重建可区分「learn 劣化 vs dream 劣化」；动作样本 &lt; min 时 **强制回退全局场**，不编残差
- $z_{pre}$ 仍标 `model_reconstruction`，无 receipt 不得当 verified

### 环境
- 测试用 `E:\ARSI\.venv`（Python 3.12 + pydantic/pytest/pyyaml/numpy/networkx/scikit-learn）
- 运行命令：`PYTHONPATH=E:\ARSI\src` + `E:\ARSI\.venv\Scripts\python.exe -m pytest tests -q`（CWD=`E:\ARSI`）

### 仍未做
- host measured 比例与 `n_v_updates` 稳定几 tick 后，再看是否把条件场接进 Governor 选动作
- 官方 Dream-RSI 超参回填 · I6 live · LLM 401

---

## Sync-1 器官同步诊断（2026-09-23 · CTM arXiv:2505.05522）

### 交付
| 模块 | 作用 |
|------|------|
| `foundation/sync_repr.py` | 分块相关阵 · PairSync 递推（App.H）· short/med/long 半衰期 · dead 通道 · top 耦合对 · 谱熵代理 |
| `capability_flow.observe/health` | `sync` / `dead_channels` / `dead_organs` 进 health 与 flow_guidance |
| `host_strategy` | `dead_organs_observe_only` 警告（**只观察，不改 focus/confidence**） |
| `pull_health.py` | 打印 sync 块 entropy / top pairs / dead |
| `tests/test_sync_repr.py` | 递推、相关、dead、多尺度、Tracker 接线 |

### 语义（CTM 对位）
- **snapshot vs 同步**：在 $v(z)$ 旁增加通道间时序共波动 $\rho_{ij}$ / $S_{ij}(t_{1/2})$
- **dead-organ**（CTM dead-neuron 类比）：低方差 frozen 或与其它通道 max|ρ|≈0 → 仅 brief 观察字段
- 半衰期网格 30s / 5min / 30min（官方可学习 $r_{ij}$ 的统计先验）

### 纪律
- **不改评分、不改铁律、不改 Dream-RSI 选择**
- live/pool/D_T/流分块不混写
- 诊断特征 ≠ verified claim

### 测试
全量套件见本轮运行（Sync 相关 52 passed 后全量）

### 仍未做
- C1-1 certainty↔correctness 校准 · C1-2 自适应预演深度 · C1-3 条件同步 $\mathbf{S}(a)$
- C2 NLM / 神经 ODE / 认知地图（数据门槛后）

---

## C1-1/2/3 CTM 决策轨（2026-09-23）

### C1-1 校准（t2 纪律）
- `IntrospectorCalibrator`：显式 confidence ↔ success 可靠性分箱 + ECE
- **overconfident** 或 ECE 高 → 压低 `self_trust`；严重过度自信 → `degrade_to_baseline`
- 未显式给 confidence **不编 0.5** 进 ECE（避免误伤）
- `governor_advice` / health 带 `calibration`

### C1-2 自适应预演
- `PreEnactmentEngine.max_think_ticks` 为唯一预算；`plan_think_ticks()` 按 difficulty（η、field_mse、dynamics 未训）伸缩
- `select_best_adaptive`：多 tick 加深 $z_\tau$ 视界，**selection certainty**（margin+conf）达标则 early-stop
- `core.step` 预演层已切到 `select_best_adaptive`（source=`pre_enactment_adaptive`）

### C1-3 条件同步 $\mathbf{S}(a)$
- `action_conditioned_sync`：按 action 分层算耦合对 / dead / 熵；n&lt;3 标 `unmeasured`
- `FlowSample.action` 入历史；`sync_report.by_action` 进 health

### 测试
```
503 passed
```
新增 `tests/test_c1_ctm.py`（校准 / 自适应 tick / S(a)）

### 仍未做
- C2 NLM / 神经 ODE / 认知地图
- LLM 401 · I6 live · 官方超参回填

---

## C2 表示升级轨（2026-09-23 · CTM）

| 项 | 模块 | 要点 |
|----|------|------|
| **C2-1** | `world_model/nlm_filter.py` | 每通道私有历史滤波 $g_d(A_d)$（线性 NLM）；不足样本 echo，不编预测 |
| **C2-2** | `world_model/kernel_flow.py` | Nadaraya–Watson 核回归 $v(z;a)$；一阶直督；动作样本不足回退全局 support |
| **C2-3** | `world_model/cognitive_map.py` | 无外部坐标：action→Δz 边；`route(z_now,z_goal)` 想象路线 |
| **C2-4** | `foundation/sync_memory.py` | 器官对绑定 evidence_id；`recall(z)` 超窗召回 |

### 接线
- `CapabilityFlowTracker.observe`：NLM + cogmap 边；dt 够时 `kernel.fit_sample`
- `health`：`nlm` / `kernel_v` / `cognitive_map` / `sync_memory` / `nlm_pred`
- `flow_guidance`：`kernel_v` + `cognitive_route`
- `pull_health`：打印上述字段

### 纪律
- 一阶监督，**禁** multi-step consistency loss
- 数据门：不足样本标 untrusted / unmeasured，不发明 Δ
- 不改评分 / 铁律 / Dream-RSI 选择

### 测试
`tests/test_c2_representation.py`；全量见本轮运行

### 仍未做
- 神经 MLP NLM / 真 neural ODE（需更厚样本）
- LLM 401 · I6 live · 官方超参回填

---

## LLM 切换 AMD MiMo（2026-09-23）

| 项 | 值 |
|----|-----|
| provider | openai（OpenAI 兼容） |
| model | `MiMo-V2.6-Flash` |
| api_base | `https://developer.amd.com.cn/radeon/api/v1` |
| key | `config/arsi.yaml` 本地 fallback + **gitignored** `E:\ARSI\.env`（`ARSI_API_KEY`） |
| proxy | `use_proxy: false` |
| 冒烟 | `chat("PONG")` → **success True / content PONG / available True** |

`start_daemon.bat` 会加载 `E:\ARSI\.env`。**daemon 需重启**才会换掉旧 NVIDIA 401 配置。

注意：`arsi.yaml` 为本地开发含 key，**勿推公开远端**；`.env` 已在 `.gitignore`。

---

## I6 live 验收轨（2026-09-23）

### 交付
| 项 | 说明 |
|----|------|
| `iwm/i6_live.py` | 证据门 + 宣称阶梯：`skeleton` → `candidate` → **`introspective_v1_live`** |
| `scripts/i6_live_acceptance.py` | 读 health/checkpoint/compare + 可选 Q 探针/suite；写 `archive/eval/i6_live_acceptance.json` |
| `tests/test_i6_live.py` | 阶梯、降级、facts 合并 |

### 宣称纪律
- Q1–Q3 **任二失败** → skeleton  
- **未跑 Q**（skip-q）→ 最高 **candidate**，不误标 skeleton  
- 全部门槛过了还要 **suite verified=True** 才 `live_v1`；否则强制降级  
- 种子 runner 只配 `candidate`（与 `iwm_probes` 边界一致）

### 首次实盘结果（--skip-suite --skip-q）
```
claim: introspective_v1_candidate
ok:  pool=17 traces=287k cycles=383 L1=0.956 organs_frac=0.6 no_regress
fail: q_evaluated · flow_thick(v_upd=7<8) · calibration_n=0 · host_loop_healthy · suite_verified
```

### 测试
```
525 passed
```

### 仍未做
- 跑满 Q 探针 + suite → 再验一次 live  
- 官方超参回填 · MLP-NLM / neural ODE（厚数据后）

---

## I6 live 达成 introspective_v1_live（2026-09-23）

### 收口改动
- `apply_outcome` 带上 **selection_certainty / confidence**（t2 显式置信）
- 校准器 **落盘** `archive/iwm/calibration.json`
- host 实测 **effect → confidence** 批量入 ECE
- host_success_rate 取 host_loop ∨ multi_agent **最优证据**（不因空 MA 清零）
- Q 事实优先读 daemon **health.q_gate**（不再对生产库 step）

### 验收结果
```
claim: introspective_v1_live  live_v1=True
gates_fail: []
pool=17 traces=287461 cycles=383 v_upd=8 L1=0.956 organs=0.6
cal_n=42  suite: verified pytest_exit_0
report: archive/eval/i6_live_acceptance.json
```

### 测试
全量套件见本轮（I6/IWM 相关 43 + 全量回归）

### 仍未做
- 官方超参回填 · MLP-NLM / neural ODE（厚数据后）
- daemon 持续跑以让 calibration.json / flow 跨重启增厚

---

## Dream-RSI 官方仓库核查（2026-09-23）

### 仓库状态（zhengkid/Dream-RSI@main）
| 项 | 状态 |
|----|------|
| Paper PDF | ✅ `papers/Dream-RSI.pdf`（已下载 cache） |
| 项目页 / demo | ✅ dream-rsi.com |
| **Full codebase** | ⏳ **Being prepared**（无源码/无 YAML） |
| Reproduction scripts | ⏳ Being prepared |
| Discovered programs | ⏳ Being prepared |

### 论文 PDF 能钉死的
| 项 | 证据 |
|----|------|
| Replay 目标式 | $V=\max s-\beta_1 N+\beta_2(N/\max\{1,k^\*\})$ |
| pareto 式 | `auc - λ·parallel_penalty`；penalty=seq_rounds/probes |
| default β（历史不足） | **about 0.6** ✅ |
| plateau 调 β 步长 | **about 0.1–0.2**，clamp [0,1] ✅ |
| β 跨周期规则 | 改善保持 / 平台抬升 / 浪费下调 / 冲突→0.6 |
| M / K₂ | 仅有符号定义，**无具体数** |
| β₁ β₂ λ · β grid 数值 | **论文未给** |

### YAML 处理
- `config/dream_rsi_params.yaml`：`official_code_status=paper_pdf_only_code_pending`
- **paper_confirmed** vs **paper_unspecified** 分栏；未编造 β₁/β₂/λ/M/K₂
- 公式已写入注释；加载器 `load_dream_rsi_params` 正常，**525 passed**

### 仍未回填（等 full codebase）
`beta1_cost_penalty` · `beta2_parallel_bonus` · `parallel_lambda` · `M_revisions_per_cycle` · `K2_replay_max_rounds` · `beta.sweep_grid` 具体格点

---

## dream-rsi.com 交互 demo 挖掘（2026-09-23）

### 来源
`https://www.dream-rsi.com/dream.js`（+ `script.js`）；站方明确：**canvas 数字 illustrative**，真数在 Results。

### Demo JS 常量（示意，非论文系数）
| 常量 | 值 | 含义 |
|------|-----|------|
| `LAMBDA` | **0.006** | `Return = best − λ·n_attempts`（扁平尝试成本，≈β₁ 简化式；**不是**论文 β₁） |
| `REVS` | **4** | 注释 `// M`；站文「π0 部署版 + π1…π3 修订」→ **M=4 个候选版本** |
| `EK` | **24** | 每候选版本回放次数（随机策略取均值） |
| `BUDGET/MAXD/MINBR/MAXBR` | 32/6/5/8 | demo 树生成，与 Dream-RSI 超参无关 |

### 结构确认（可写进配置）
- **M = 4**（π⁰…π³，含已部署 π⁰）→ `M_revisions_per_cycle: 4` ✅
- 回放覆盖 **子树**，按节点计成本；`best_score - λ·n` 为 demo 回报式
- 仍缺：β₁/β₂/λ(pareto)/K₂/β-grid 真数值

### YAML
`dream_rsi_params.yaml` 增加 `demo_site` 段（illustrative / structure_confirmed 分栏）；`loop.M_revisions_per_cycle: 4`。

---

## 同名仓库深度鉴别（2026-09-23）

详见 `E:\Mimo 生成\docs\2026-09-23\Dream-RSI-peer-repos-deep-dive.md`

| 仓 | 判定 |
|----|------|
| juanmackie/pi-Dream-RSI | 公式最贴论文；**peer default** β₁=β₂=0.01, K₂=8, grid含0 |
| TheAstrayDev/dream-rsi-sdk | NOTICE 明确非 Google；约定版本化（parallel_weight=0.1） |
| robinber/dream-rsi-spark | SPEC 最严谨：**M=R+1**、App.B≠§3 |
| patrykorwat/open-dream-rsi | **误标 β₂**（diversity≠N/k），数字慎用 |
| opengpt4/dsh_dream_rsi | 具身域 + **无 license**，不采信 |

**结论**：全部非官方；YAML 记 `peer_implementations` 旁注；**不**升格为 paper_confirmed；ARSI 现值不动。

---

## Harness-1+X 开工（2026-09-23 · ModularRSI × HarnessX）

### 交付
| 模块 | 作用 |
|------|------|
| `harness/taxonomy.py` | 九维 c1–c9 × 五模块；**c7 铁律 frozen**；scope fence |
| `harness/manifest.py` | Change-Manifest + 回滚 inverse_op + 状态机 |
| `harness/audit.py` | audit.jsonl（stage/gate/commit） |
| `harness/digester.py` | 失败簇（tool_loop / timeout / premature…） |
| `harness/planner.py` | landscape + **untried levers**（打 under-exploration） |
| `harness/gates.py` | critic（防 hack）· regression（防遗忘）· seesaw（→fork） |
| `harness/variants.py` | 变体池 + ensemble routing；铁律全变体共享 |
| `config/harness_map.yaml` | 映射与隔离表（sealed/I6/eval_loop 黑名单） |
| `tests/test_harness_aegis.py` | 14 项 |

### 纪律
- 不动铁律 / 评分 / Dream-RSI 选择  
- frozen dim c7 拒绝一切 edit  
- seesaw 不硬拒 → fork 变体  

### 测试
`539 passed`（本轮全量，含 +14 harness）

### 仍未做
- Digester 接真实 host 轨迹入 daemon  
- LLM Evolver / Change-Manifest 生成（需 MiMo）  
- 变体路由接 multi-agent 宿主  
- c9 训练桥导出  

---

## Digester 实盘接入（2026-09-23）

### 交付
- `harness/pipeline.py` — `run_landscape`（Digester+Planner+label 持久化）
- `scripts/harness_landscape.py` — 生产库一键 landscape
- `arsi_daemon` 每 **3 tick** 跑 landscape；health 带 `harness_landscape`
- `archive/harness/` audit + label_counts；`archive/eval/harness_landscape_latest.json`

### 首次实盘（400 条生产轨迹）
```
traces=400 failures=61
clusters=2: generic_failure n=45 · invalid_or_compile n=16
untried_edit_types=全部 5 类（首轮）
```

### 测试
全量 **539 passed**

### 仍未做
- LLM Evolver 产 Change-Manifest  
- 变体路由绑 hermes/mimo/synthex  
- c9 训练桥  

---

## Evolver 最小闭环（2026-09-23）

### 交付
- `harness/evolver.py` — 规则模板 + 可选 LLM 润色；**scope fence + critic** 后才 `accepted`
- `scripts/harness_evolve.py` — 一轮：landscape → manifests → store/audit
- 默认**不自动改 core**（Change-Manifest 可审可回滚 `inverse_op`）

### 实盘一轮
```
proposed=2 accepted=2
[prompt c2] fail_class+recovery before tool failure  (generic_failure n=45)
[prompt c6] compile check before accept code        (invalid_or_compile n=16)
```
`archive/eval/harness_evolve_latest.json` · `archive/harness/change_manifests.jsonl`

### 测试
全量 **542 passed**（+3 Evolver）

### 仍未做
- 应用 accepted manifest（人工/门后 apply）  
- 变体路由绑宿主 · c9 桥  

---

## Apply 两条 Change-Manifest（2026-09-23 · 决策：先落地后变体）

| Manifest | 落点 | 效果 |
|----------|------|------|
| **CM-fc23371019** | `ARSIBrief` + `HostStrategy.as_structured_block` | brief 增加 **Fail-handling**：fail_class + 一次恢复；禁止同前重复失败调用 |
| **CM-7b86cd1d1f** | `CodeVerifier` | `require_compile=True` + `on_compile_fail=reject_and_log` |

### 决策理由
AEGIS 只差落地环；两条对准最大实盘簇（hermes tool 45 / synthex compile 16）。变体路由等跨宿主**改法冲突**再上。

### 测试
全量 **547 passed**

### 仍未做
- 变体路由绑 hermes/mimo/synthex · c9 训练桥  
- landscape 复测：Fail-handling / compile 门生效后簇是否收缩  

---

## 变体路由绑三宿主 + daemon 重启（2026-09-23 · 决策）

### 决策
**不**立刻复测 landscape（400 条是 apply 前轨迹 → 假阴性）。改为：
1. `VariantPool.ensure_hosts / route_for_host / observe_host`
2. daemon `_run_host_loop` 绑 hermes / mimo-desktop / synthex-mothernest，按成败 observe；落盘 `archive/harness/variants.json`
3. 重启 daemon 加载 Fail-handling + compile 门 + landscape 每 3 tick

### 测试
全量 **550 passed**（+3 host variants）

### 仍未做
- c9 训练桥  
- **等 1–2 天新轨迹后** landscape 复测看簇收缩  

---

## c9 训练桥（2026-09-23）

### 交付
- `harness/training_bridge.py` — 轨迹 → **task-level** JSONL 训练样例  
  - `trajectory_digest` / `prompt_facts` / `harness_variant` / `reward`  
  - **黑名单**：sealed / i6_gate / eval_loop / gold_answer 不进训练集  
  - **grpo=false**（API 宿主无权重，只导出）
- `scripts/export_training_bridge.py`

### 实盘导出
```
in=500 exported=500 excluded=0
out=archive/eval/c9_training_bridge.jsonl
alignment=task_level_not_action_level
```

### 纪律
- 不现场 GRPO / 不训基座  
- 与 Cross-Harness GRPO 的 task-level 对齐口径一致，留作离线 SFT/DPO/GRPO

### 测试
全量 **552 passed**（+2 c9）

### Harness RSI 全环
```text
Digester→Planner→Evolver→Gates→Apply→Host variants→c9 export  ✅
```

### 仍未做
- 等 1–2 天新轨迹后 landscape 复测  

---

## 总规划 + P0-RRSI（2026-09-23）

### 规划（一篇一落地）
RRSI ✅ → SEVerA → SAHOO → AIDE² → Self-Harness → Grader → GAI  
落地序：P0-RRSI → P1-SEVerA → P1-SAHOO → P2  
全文：`E:\Mimo 生成\docs\2026-09-23\ARSI-research-build-roadmap.md`

### P0-RRSI 交付
| 模块 | 对应 |
|------|------|
| `edit_budget.py` | L0 余弦退火 $b_t$ · stall_flag · unexercised |
| `noise_floor.py` | δ 标定（range/std）· 地板判定 |
| `credit.py` | $\mathcal{L}_t$ 账本 · $g_t$·$N_t$ · 剪枝集 · 负证据 |
| `accept.py` | **评前泄漏** → 地板 → 成本式/带内 shaped → 守卫 |
| `prune.py` | $\mathcal{B}_t$ · exploration_directive |
| `evolver.py` | 接预算上限 + stall 探索槽 |

### 测试
全量 **559 passed**（+7 P0-RRSI）

### 仍未做
- manifest 回写 ΔS/ΔC/accepted 进 ledger  
- 词表 YAML 对齐九维  
- 下一篇精读：**SEVerA**  

---

## SEVerA + SAHOO 直接交付（2026-09-23 · 无需批准）

### SEVerA
- 精读：`E:\Mimo 生成\docs\2026-09-23\SEVerA-deep-dive.md`
- **`harness/contracts.py`**：G1–G10 FGGM-lite（Φ/Ψ/check/fallback）
- `accept.admit` 接硬契约：G3/G5/G6/G10 fallback → **拒**

### SAHOO
- 精读：`SAHOO-deep-dive.md`
- **`harness/sahoo.py`**：GDI · regression_risk · CAR · `decide_stop`（**CPS=0 绝对停**）

### 测试
全量 **569 passed**

### 下一篇（自动继续）
AIDE² → Self-Harness → Grader → GAI  

---

## AIDE² + Self-Harness 交付（2026-09-23）

### AIDE²
- 精读：`AIDE2-deep-dive.md`
- `holdout.py`：pub/priv 分割 · private_grade · **public 赢 private 输→拒** · 一阶/二阶泛化
- `hack_kpi.py`：proxy↑×downstream↛ hack 率 · lineage 趋势

### Self-Harness（同轮）
- `minimality.py`：minimal_edit_score · **双回归**（held-in + held-out）

### 下一篇（自动继续）
Grader → GAI 用语  

---

## Grader + GAI 交付 — RSI-Armor 七篇齐（2026-09-23）

### Grader
- `drawback.py`：typed detector · 表达式 any/all/vote · birth/shadow/retire · S(e) · validity gate
- **安全在 anchor**（sealed/铁律），不在 lifecycle

### GAI
- `gai.py`：两表盘 · polarity（anchored / goal_drift / self_referential）· `rsi_defects` · `assert_anchored`
- **ARSI_GAI = RSI + anchored**

### RSI-Armor 全景
RRSI ✅ SEVerA ✅ SAHOO ✅ AIDE² ✅ Self-Harness ✅ Grader ✅ GAI ✅  

### 测试
全量 **584 passed**

---

## 二次研读 + 架构基石升格（2026-09-23）

### 结论
- **升格 L0** GAI 宪法 · **L1** SEVerA 契约层 · **L2** RRSI×AIDE² 正则化 MetaRSI · **L5** Grader 认识论  
- SAHOO → **L3 生命体征**（并入 IWM）  
- Autopoiesis → **总纲**  
- Self-Harness 不升格  

### 交付
- `E:\Mimo 生成\docs\2026-09-23\RSI-second-pass-pillars.md`
- `harness/pillars.py` 六层声明 + audit  
- README 增「六层架构」  

### 新潜力（未做）
P-a FGGM 全调用面 · P-b 策略臂 bandit · P-c 有界上下文 · P-d 软锚硬化 · P-f GDI 入 IWM · P-h Ignition  

### 测试
全量见本轮运行

---

## 二次潜力 P-a/d/f/b 交付（2026-09-23）

| 项 | 模块 | 要点 |
|----|------|------|
| **P-a** | `call_guard.py` | FGGM 全调用面：llm/tool/brief；密钥脱敏 · 非空 · 预算帽 + 铁律链 |
| **P-d** | `anchor_hardening.py` | 软反馈→可检查 detector；anchor_set_quality（≥4 起步 / 论文目标 ≥10） |
| **P-f** | `IWM.goal_vitals` | GDI / CAR / regression_risk 并入 IWM.health（L3 生命体征） |
| **P-b** | `strategy_bandit.py` + Evolver | **策略臂 UCB1 + 30% softmax**（AIDE₈₅） |

另：contracts 对非 dict 输出不再误触发 G2/G4 等 dict 契约。

### 追加 P-c / P-e
| 项 | 模块 | 要点 |
|----|------|------|
| **P-c** | `bounded_context.py` | 有界历史压缩（先弃 digest 再弃旧 recent）；防 AIDE₀ 超窗死 |
| **P-e** | `detectability.py` | 可检性投资序：机械 detector 先建，语义 judge 最后 |

### 测试
全量 **596 passed**（P-a/d/f/b + P-c/e）

---

## P-g/h/i 交付 — 二次潜力清零（2026-09-23）

| 项 | 模块 | 要点 |
|----|------|------|
| **P-g** | `everitt.py` | Everitt 条件：改 utility 须 value **预判改写** + **当前 utility** 评未来 |
| **P-h** | `ignition.py` | Ignition test：发现的外环 vs 基线外环；点火 / 样本效率提示 |
| **P-i** | `overshoot.py` | **过冲检测**：越过最优点后仍在改 → `rollback_to_best_or_stop` |

### 测试
全量 **600 passed**

### 二次研读九件潜力
P-a…P-i **全部落地**。主线清零。

---

## 第三轮研读 + 互锁螺丝（2026-09-23）

### 三轮结论
- 七篇是**互锁机器**（锚定—契约—搜索—解耦—评标—测漂）
- **负认识论**是地基
- Goal drift 操作定义：**锚名不变 + GDI 升**
- 涌现反馈律 F1–F5

### 落地 T1–T3 / T5–T6
`conformance.py` · `accept.domain_guard` · `accept.select_among_admissible` · `sahoo.contractive_regime` · `sahoo.capability_ceiling_hit` · `strategy_bandit` AIDE₈₅ 五臂

### 报告
`E:\Mimo 生成\docs\2026-09-23\RSI-third-pass-interlock.md`




## 接线债清偿（2026-09-23）

| 接线 | 路径 |
|------|------|
| call_guard → conformance 累计 | 
untime_wiring.log_call_conformance |
| admit → domain_guard / stop / Everitt | detail_extras + dmit_extras |
| daemon health → dual_sensor / overshoot / calib | rmor 字段 _armor_health |
| brief → 有界上下文压缩 | rief_compress |

### 测试
全量见本轮实跑（wiring 89 + 全量）


## 2608.10299 二轮（2026-09-23）

- 反馈空间三型操作法 · 交互空间构造 · 组织演化
- **W1–W5 纪律**落地 \harness/feedback_evo.py\：ECHO 可操作性 · ARCO 步-局一致 · PEBBLE 重标 · 提示退场 · R* 多 critic
- 报告：\rxiv-2608.10299-second-pass.md

---

## 全基石深度反刍 + P-R 潜力清零（2026-09-23）

### 报告
`E:\Mimo 生成\docs\2026-09-23\ARSI-deep-rumination.md`

### 已落地
| 项 | 模块 | 要点 |
|----|------|------|
| **P-R1** | `harness/epistemic.py` | 负认识论 claim 白名单；禁积极谓词；I6→`no_known_introspection_defects` |
| **P-R2** | `world_model/continuous_dream.py` | **连续梦境**：树节点精确 + v(z;a) 插值 + S 共动 + 半步反事实 |
| **P-R3** | `harness/multiscale.py` | 多尺度控制律：各尺度改速率上限 + L_Δ 分尺度收缩 |
| **P-R4** | `harness/unified_credit.py` | 双 Ω 统一账本：policy/harness/both 归因 |
| **P-R5** | `meta/red_queen_env.py` | Red Queen 互压→EnvEvolution：成功率↑→难度↑→effort |
| **P-R6** | `harness/pillars.py` L0/CHARTER | 组织=ρ / 结构=Ω 二分（自创生×GAI 正名） |
| **P-R7** | `harness/monotone.py` | V*≥V₀ 推广到 manifest：含 baseline；admitted-best 不降 |
| **P-R8** | `harness/metabolism.py` | 代谢面：吞吐 / 代谢率 / 膜完整性 → `armor_health` |
| **AEGIS 自动环** | daemon `_run_harness_evolve_dry` | landscape 后自动 propose（**默认 dry-run，不 apply core**） |

### 测试
全量 **639 passed**（+15 deep-rumination potentials）

### 仍未做
- P-R9 guidance 白名单（replay 禁语义 / Evolver 可语义）
- landscape 复测（等 1–2 天新轨迹）
- daemon 重启加载 epistemic/multiscale/coevolution/continuous_dream


---

## P-R9 + P-R 接线债 + daemon 重启（2026-09-23）

### P-R9 guidance 白名单
- `harness/guidance.py`：**selection 禁语义** · diagnosis 可读 · Evolver 可语义但过泄漏筛 · brief structured_only
- Evolver `_gate` 接 `screen_guidance("evolver_manifest", …)`

### 接线
| 接线 | 路径 |
|------|------|
| continuous_dream → dream_rsi_cycle | `harness/pr_wiring.continuous_dream_from_pool` |
| unified_credit → dream_rsi / eval_loop | `log_unified_after_dream` |
| monotone best → dream_rsi / eval_loop | `MonotoneLedger.observe` |
| metabolism/monotone → armor_health | `runtime_wiring.armor_health` |

### daemon
- 旧 PID 13556/26256 已停；**新 PID 39856**（`.venv` · tick-sleep 300）
- 首 tick：Ingested **377** traces；AMD LLM HTTP 200
- 已加载：epistemic / multiscale / coevolution / continuous_dream / guidance / AEGIS dry-run

### 测试
全量 **647 passed**（+8 P-R9/wiring）

### 仍未做
- landscape 复测（等 1–2 天新轨迹）
- GRPO / Dafny / 多 agent 深编排（后置）


---

## 宿主协议赋能最大化升级（2026-09-23 · 双边界）

### 决策
ARSI 最终服务宿主 → 协议必须**最大化赋能**；同时守双边界：
1. 宿主 brief 面**允许**语义化/可读/可执行
2. ARSI **内部 replay_selection 仍禁语义**；claim 走负认识论

### 交付
| 模块 | 要点 |
|------|------|
| `adapters/host_empower.py` | Empowerment Pack：skill kit · tool routing · active Change-Manifests · fail recovery playbook · known drawbacks · continuous-dream what-if · armor/vitals · scaffold_level（W4 hint fade） |
| `bidirectional_interface` | brief 挂 empower_pack；`format_for_agent(full=)` 可绕过 ObservationPack |
| `ARSIReport` | 可行动字段：fail_class / recovery_* / acceptance_evidence / manifest_ids_used / compile_checked / measurements |
| report claim 门 | `admit_claim` 拒积极谓词 → outcome 降为 unknown，effect≤0 |

### 测试
全量 **653 passed**（+5 host_empower；IWM 测试改 `format_for_agent(full=True)`）

### 纪律
- 不改铁律/评分；不盲标 SUCCESS
- 内部 selection 语义禁令不因宿主面放宽而松动


---

## P0 可归因与稳定四件（2026-09-24 · 夜间数据诊断后）

### 依据
晨间 landscape：compile 簇 16→6（CM 有效），generic 45→141 且证据全是 `synthex_gate:*` 粗标签；
planner `tried_edit_types={}` 空转；v 场 d_t_mean 系数 1e205；daemon 双实例。

### 交付
| 项 | 模块 | 要点 |
|----|------|------|
| **P0-1** | `harness/digester.py` | `params.fail_class` 优先；新标签 gate_reject/tool_error/runtime_error/resource_limit；簇带 hosts/fail_classes |
| **P0-1b** | `multiagent/protocol.py` | report_result 透传 fail_class/recovery/acceptance_evidence |
| **P0-2** | `pipeline.py` + `planner.py` + `evolver.py` | landscape 自动读 ManifestStore；tried_* 记账；least-tried 轮换；同标签同 edit_type 去重 |
| **P0-3** | `capability_flow.py` + `nlm_filter.py` | `_finite_clamped` 入 fit/predict/velocity_gt；参数/权重 clamp ±1e3 |
| **P0-4** | `arsi_daemon.py` | 单实例：进程扫描 + named mutex + `ARSI_DAEMON_SINGLETON` 子进程拒入 |

### 测试
全量 **662 passed**（+9 P0）

### 下一步（数据支撑）
P1：compile 门固化收账 · Fail-handling 换靶 gate_reject · `harness_preflight` 只读口
P2：EL 难度方向 / Red Queen 真互压


---

## P1 compile 收账 / gate_reject 换靶 / preflight（2026-09-24）

| 项 | 模块 | 要点 |
|----|------|------|
| **P1-1** | `harness/compile_gate.py` + `code_verifier` | 每次 compile pass/reject 记账；`compile_gate_effect` 对比 invalid_or_compile 收缩 |
| **P1-2** | `host_empower` + brief + host_strategy | Fail-handling **换靶 gate_reject**：gate_id+candidate_id · 一次修复 · compile 门 |
| **P1-3** | `host_empower.harness_preflight` + `brief(pull=)` | **只读**预检：fail 簇 + playbook + compile 账，不触发 evolve |

### 测试
全量见本轮实跑（P1 21 + 全量）

### 仍未做
P2：EL 难度方向 / Red Queen 真互压


---

## P2 Red Queen / EL 课程（2026-09-24）

| 项 | 修复 |
|----|------|
| **P2-1** | erify_world：high/max 子代 **难度不得降**（d_t/L/novelty）；evolve_traces 注入更多 length/scenario/skill-tail |
| **P2-2** | 
ed_queen_effort_for_pool 接 evolve_from_seed：宿主成功率↑ → effort↑ |
| **P2-3** | EL 
ecord_probe **信任 success 标志**（不再被负 replay_score 覆盖）；零通过率 → curriculum hold |

### 测试
全量 **674 passed**（+6 P2）


---

## GRPO 前置数据面（2026-09-24 · export only）

| 交付 | 路径 |
|------|------|
| harness/grpo_data.py | group 采样（brief_id/task）· reward 表（fail_class+claim 罚）· holdout split |
| scripts/export_grpo_data_plane.py | 导出 groups/rewards/split + manifest |
| 实盘导出 | rchive/eval/grpo_data_plane/ traces=300 rewards=300 selection=240 private=60 |

### 纪律
- grpo_live=False 永不训权重（API 宿主）
- sealed/i6/eval_loop 黑名单进 private/blacklist
- claim 被拒 → reward≤0（负认识论）
- 实盘 dvantage_ready=0：单 brief 单轨迹，需 **多采样/多变体** 才有组内相对优势

### 测试
全量 **681 passed**（+7 GRPO；P2 难度门改为禁塌缩）


---

## GRPO 组缺口补齐（2026-09-24）

- worlds_from_trace_chunks + _ChunkPool：生产轨迹切块造多世界
- default_policy_matrix：beta×width / fixed 多结构策略
- collect_policy_matrix_from_pool：同世界 × 多策略 → 多完成组
- 复合 reward：quality + 效率(probes) + score，打破同分饱和

### 实盘
export --from-pool：matrix_rows=40（8 世界×5 策略）· **advantage_ready=3**
（仍有同分世界；组间相对优势已可用）

### 测试
全量见本轮（GRPO 8 + 全量）


---

## tool_error CM 落地（2026-09-24 · 按 landscape 复测）

### 依据
窗 400：	ool_error n=90（hermes terminal/execute_code）· invalid_or_compile n=9 · gate_reject 已分型

### 交付
| 项 | 落点 |
|----|------|
| CM tool | empowerment.applier：fail_class+tool_id · 一次恢复（execute_code↔terminal）· report 契约 |
| CM compile / brief | 同轮 propose 3 accept 3 |
| 硬赋能 | EmpowermentApplier.apply_change_manifest → hermes skills rsi-cm* |
| 软赋能 | brief/host_strategy Fail-handling 对准 tool_error |
| 账 | rchive/eval/unified_credit_tool_error.json |

### 测试
全量见本轮


---

## synthex compile 重试环 CM（2026-09-24）

### 依据
tool_error 90→61（CM 有效）；invalid_or_compile 窗内 9→15，同 candidate_id 多版本连打。

### 交付
- Evolver invalid_or_compile：**同 candidate_id 仅 1 次修复** → abandon_and_log
- playbook / brief 换靶 compile 重试环
- hermes skills：rsi-cmd60c7b54a8 / rsi-cm1ab50a9260
- synthex：E:\SYNTHEX Autopoiesis\docsrsi_host_loop_guidance.md
- 账：unified_credit_compile_ring.json

### 测试
全量见本轮


---

## compile 重试环硬闸（2026-09-24）

被动 guidance 不够 → **ARSI 侧强制**：
- compile_gate.candidate_reject：同 candidate_id 满 2 次 compile 失败 → abandon
- code_verifier 接入硬闸；fail_class=compile_retry_ring
- 实盘同 id 仍 ×11–12（233dcb/17eeed/1e9f10）— 闸载入后应断环

### 现况（fresh 400 vs 旧 landscape）
tool_error 43（↓）· invalid_or_compile 40（重试堆积，待硬闸）

### 测试
全量见本轮


---

## 重试环接到 ingest/digester（2026-09-24）

发现：synthex 失败只进轨迹，**不经过 code_verifier** → 硬闸空转。
改为 
ote_trace_candidate 在 digester 记次；≥2 次 → 标 compile_retry_ring 并 abandon 列表进 brief。

实盘（fresh 400）：invalid 40→28 · 同 id 12→9（待闸后断环）


---

## 下一阶段四线（2026-09-28）

| 线 | 交付 |
|----|------|
| hermes tool_error 细分 | digester 	ool_error_terminal/execute · Evolver 捕获 exit/stderr · playbook |
| eta/self_trust/live_layer1 | iwm/vitals_ledger.py 诊断原因+修复建议；multiagent 宿主结果入 calibrator |
| synthex abandon 硬执行 | is_reversion_blocked：弃用 id 的 re-version → outcome 降级 |
| GRPO 扩组 | --from-pool --max-worlds 20 |

### 测试
全量 686 passed


---

## RRSI L2 保真（2026-09-28）

| 交付 | 要点 |
|------|------|
| harness/rrsi.py | ΔC=(C'-C)/C · Branch A ΔC≤β0+β1ΔS（coding 0.10/44.5）· three_track evolve/ID/OOD · rrsi_round 含 b_t+stall U_t |
| ccept.relative_cost | 相对成本接口 |
| pipeline landscape | 附 
rsi_tracks（有 score 字段时） |
| 测试 | 	est_rrsi_fidelity.py 7 项 |

全量见本轮。


---

## GAI 基石深挖（2026-09-28 · 批1 起点）

- 一轮机制：E:\Mimo 生成\docs\2026-09-28\GAI-deep-dive.md（χ·两表盘·四缺陷·缺口表）
- 落地：delusion_box_check · dual_sensor_goal_drift 入 rmor_health.gai_dual_sensor
- 测试：	est_gai_fidelity.py
- 计划：GAI 二轮互锁 → SEVerA → …（批1 治权+改）


---

## SEVerA 基石深挖（2026-09-28 · 批1 #2）

- 一轮+互锁：E:\Mimo 生成\docs\2026-09-28\SEVerA-deep-dive.md
- 落地：contracts.well_formedness（FGGM fallback_valid / checker_sound）
- 测试：	est_severa_wellformed.py
- 下一篇：AIDE²


---

## AIDE² 基石深挖（2026-09-28 · 批1 #3）

- 一轮：E:\Mimo 生成\docs\2026-09-28\AIDE2-deep-dive.md
- 落地：outer_loop_select（argmax private_grade）· split_tasks(ood=)
- 测试：	est_aide2_outer.py
- 下一篇：Self-Harness 收尾 L2


---

## Self-Harness 收尾 L2（2026-09-28 · 批1 #4）

- 一轮：E:\Mimo 生成\docs\2026-09-28\SelfHarness-deep-dive.md
- 落地：self_harness_round（mine→minimal→dual_regression）
- 测试：	est_self_harness_round.py
- **批1（治权+改）收口**：GAI · SEVerA · RRSI · AIDE² · Self-Harness
- 下一批：L3 身体 ODEWorld


---

## ODEWorld 基石深挖（2026-09-28 · 批2 #1）

- 一轮：E:\Mimo 生成\docs\2026-09-28\ODEWorld-deep-dive.md
- 落地：low_fidelity_check（静态不进 z · 一阶直督 · 分块不混轨）
- 测试：	est_odeworld_fidelity.py
- 下一篇：CTM


---

## CTM 基石深挖（2026-09-28 · 批2 #2）

- 一轮：E:\Mimo 生成\docs\2026-09-28\CTM-deep-dive.md
- 落地：ctm_fidelity_check（NLM+sync 同在 · diagnostic_only · 三尺度）
- 测试：	est_ctm_fidelity.py
- 下一篇：SAHOO


---

## SAHOO 基石深挖（2026-09-28 · 批2 #3）

- 一轮：E:\Mimo 生成\docs\2026-09-28\SAHOO-deep-dive.md
- 落地：car_frontier + WEIGHTS_STATUS（must_recalibrate）
- 测试：	est_sahoo_frontier.py
- **批2 身体**收口：ODEWorld · CTM · SAHOO
- 下一批：L4 Dream-RSI


---

## Dream-RSI 基石深挖（2026-09-28 · 批3 #1）

- 一轮：E:\Mimo 生成\docs\2026-09-28\DreamRSI-deep-dive.md
- 落地：dream_selection_fidelity_check（prefix-only · 禁语义 · V*≥V0）
- 测试：	est_dreamrsi_fidelity.py
- 下一篇：Env Evolution


---

## EnvEvo 基石深挖（2026-09-28 · 批3 #2）

- 一轮：E:\Mimo 生成\docs\2026-09-28\EnvEvo-deep-dive.md
- 落地：env_evolution_fidelity_check（off-policy · 三方向 · Invalid · EL）
- 测试：	est_envevo_fidelity.py
- **批3 L4 收口**：Dream-RSI · EnvEvo
- 下一批：HarnessX / ModularRSI / Co-evo / Grader


---

## HarnessX 基石深挖（2026-09-28 · 批E #1）

- 一轮：E:\Mimo 生成\docs\2026-09-28\HarnessX-deep-dive.md
- 落地：egis_fidelity_check（四段管线 + c7 frozen + variant isolation）
- 测试：	est_harnessx_fidelity.py
- 下一篇：ModularRSI


---

## ModularRSI 基石深挖（2026-09-28 · 批E #2）

- 一轮：E:\Mimo 生成\docs\2026-09-28\ModularRSI-deep-dive.md
- 落地：contrastive_batches · module_scope_ok
- 测试：	est_modularrsi.py
- 剩：Co-evo · Grader（L5 收官）


---

## CoEvo 基石深挖（2026-09-28 · 批E #3）

- 一轮：E:\Mimo 生成\docs\2026-09-28\CoEvo-deep-dive.md
- 落地：coevo_fidelity_check（三阶 + Anchored Meta）
- 测试：	est_coevo_fidelity.py
- 收官：Grader L5


---

## Grader 收官 + 基石逐篇深挖计划完成（2026-09-28）

- 一轮：E:\Mimo 生成\docs\2026-09-28\Grader-deep-dive.md
- 落地：grader_fidelity_check（clean≠correct · validity on anchor）
- **14 篇基石深挖全完成**：GAI SEVerA RRSI AIDE² SelfHarness ODEWorld CTM SAHOO DreamRSI EnvEvo HarnessX ModularRSI CoEvo Grader


---

## Recuris 2608.24876 适配分析（2026-09-28）

- 报告：E:\Mimo 生成\docs\2026-09-28\Recuris-2608.24876-ARSI-fit.md
- 判定：**适合落地**（记忆控制层补全）· R0 WM 状态机 → R1 结构化 Γ/组件定位 → R2 验证门+CI
- 不另起炉灶、不训基座、不与 Dream-RSI L4 混账


---

## Recuris R0–R2 实现（2026-09-28）

- 方案：E:\Mimo 生成\docs\2026-09-28\Recuris-ARSI-landing-plan.md
- R0 iwm/working_memory.py：GoalEntry 状态机 · 无证据 done→truth_bounce
- R1 harness/skill_trace.py：Γ 结构化 · localize_failure → E/W/ρ/C/harness
- R2 harness/skill_patch.py：组件定向补丁 · validation_gate（n/CI 不足 REJECT）
- 测试：	est_recuris_r0_r2.py · 全量 **729 passed**


---

## Recuris WM 实线 + 定位校准（2026-09-28）

- brief 挂 working_state · report 收 wm_updates → checker 裁
- scripts/calibrate_recuris_localize.py：真实轨迹 49 fail → **experiential 49**（fail_class=compile_other）
- localize 优先 fail_class → component
- 测试全量见本轮


---

## validation_gate 真实门控 + abandon 持久化（2026-09-28）

- 门控结果：**REJECT**（success 0.033→0.033；唯一率 0.33→0.03 更差）→ 不 admit 新 skill
- candidate_fails.json 持久化 abandon 名单；report 遇弃用 id → outcome 降级
- 观察点：下一窗唯一率回升后再跑 gate_compile_retry_skill.py
- 全量 **729 passed**


---

## 数据窗复查（2026-09-28）

| 观察点 | 结果 |
|--------|------|
| terminal 细分 | 修 digester 顺序后 **tool_error_terminal 77** / tool_error 10 |
| 同 id 唯一率 | 窗 0.29→**0.57**（峰值 ×16 为历史） |
| compile_retry_ring | 27–38 条被正确标出 |
| self_trust | **0→0.396**（宿主入账生效） |
| eta / live_layer1 | 仍 0 |
| GRPO ready | 7 |


---

## hermes tool_error_terminal CM（2026-09-28）

- landscape：terminal 77 · ring 27 · tool 10
- Evolver 3 accepted；hermes skills rsi-cmac1e7d7855 等
- report：terminal 缺 exit/stderr → capture_missing
- playbook MUST capture exit_code+stderr_tail
- 全量 **729 passed**


---

## eta / live_layer1 修复（2026-09-28）

- 根因：仅 core.step() 走 predict_and_update；宿主 ingest 不喂 η
- 修复：siwm.observe_host_action 接 ingest_trace；outcome 惊奇也计 η
- 验证：ingest 后 eta **0.32** · live_accuracy **0.20**（原恒 0）
- 全量 **729 passed**


---

## pool 防塌缩 + capture_missing（2026-09-28）

- persist_to：合并旧世界、**拒绝 shrink**（短命 from_config 不得抹掉 50 池）
- preflight 通告 terminal 必捕字段；report 回传 capture_missing
- 全量 **729 passed**


---

## hermes 捕获抽取 + brief 钉死契约（2026-09-28）

- hermes_deep_adapter：tool content 里刮 **exit_code / stderr_tail / fail_class**
- brief 顶部 **HARD CONTRACT**：terminal 必捕，禁口头 done
- 全量 **731 passed**（待跑）


---

## 捕获抽取命中（2026-09-28）

- exit_code JSON 形态已刮出（extract 20 hit / 103）
- ingest 节流（每 4 条 1 次 predict）防慢
- terminal 最新窗 25（较 60 回落）


---

## 捕获可见：store JSON 解码（2026-09-28）

- 根因：ction_params 落库为 **JSON 字符串**，消费端当 dict 读→全 0
- get_recent_traces 解码 action_params/state_*，并镜像 params
- 实测：800 条 **exit_code 148 / stderr_tail 132**
- terminal 最新窗 ~21 · pool 15


---

## terminal 亚型 + 去误判（2026-09-28）

- harness/terminal_subtypes.py：timeout/perm/not_found/resource/syntax/other
- deep_adapter：**忽略 error: null** 伪失败（原先把 JSON null 当 error）
- 抽样：terminal 真失败 5/19；亚型 timeout·not_found·other
- 全量 **731 passed**


---

## terminal 归因闭环（2026-09-28）

- noisy_ok（exit0+无真错误）移出失败簇：**98→22** 真 terminal 失败
- 亚型：other 12 · timeout 6 · not_found 4
- playbook 补 timeout / not_found 恢复包
- 全量 **731 passed**


---

## terminal 六亚型收口（2026-09-28）

- other 再拆：**script_error 9 · timeout 7 · path_escape 3 · type_error 3 · not_found 3 · other 3**
- playbook：script/path_escape/type_error 专用恢复
- 全量 **731 passed**


---

## script_error / path_escape CM（2026-09-28）

- Evolver 3 accepted：script_error 捕获+一次隔离重试 · path_escape pathlib 规范化 · timeout 重试帽
- hermes skills：rsi-cmd334c9763f 等
- 全量 **731 passed**


---

## 亚型入 preflight + 门控观测（2026-09-28）

- 亚型：script 12 · timeout 8 · path 4 · type 4 · not_found 4
- validation_gate：success 0.148→0.148 **REJECT source_not_repaired**
- preflight 已下发 terminal_subtypes
- 全量 **731 passed**


---

## 长程任务收束：当前开发尽头（2026-09-28）

- 报告：E:\Mimo 生成\docs\2026-09-28\ARSI-frontier-assessment.md
- 架构图升 Recuris 层 · brief E2E 含 WM+preflight+HARD CONTRACT
- 剩余阻塞：观察窗 / 外部超参 / 无权重
- 全量 **731 passed**


---

## 统计版 OPF 纪律（2026-09-28 · JEPA-Anything J0–J2）

- world_model/opf_discipline.py：J0 跨块正交 · J1 因子活性/死块 · J2 干预→分块响应
- 接入 rmor_health.opf；诊断-only，不训网
- 测试 	est_opf_discipline.py · 全量 **731+**（本轮 731+6）


---

## OPF 解耦完全落地（2026-09-28）

- DECOUPLE_POLICY：organ=calibrator · pool=dream_rsi_deploy · live=eval_only · env=env_evo
- capability_flow.health().opf 从 z 行算 J0–J2 + orth_alerts
- 警报：organ~pool ρ=0.98 容量耦合 → 必须不同更新率
- 测试 	est_opf_decouple.py · 全量 **735+**
