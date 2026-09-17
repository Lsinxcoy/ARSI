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
