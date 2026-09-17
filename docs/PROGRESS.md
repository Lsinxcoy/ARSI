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
