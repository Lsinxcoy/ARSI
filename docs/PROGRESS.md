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

