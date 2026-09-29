# ARSI 生产运行手册（Runbook）

**版本**: 2026-09-29 · **目标**: 完全投产

---

## 1. 启动 / 停止

```powershell
# 启动（推荐）
powershell -File E:\ARSI\scripts\ops_start.ps1

# 停止
powershell -File E:\ARSI\scripts\ops_stop.ps1

# 健康探针
powershell -File E:\ARSI\scripts\ops_health.ps1

# 状态备份
powershell -File E:\ARSI\scripts\ops_backup.ps1
```

单实例：`O_EXCL` 锁 + 会话互斥；**venv 代理壳 + 基座解释器 = 一个逻辑 daemon**。

---

## 2. 配置与密钥

| 项 | 位置 | 规则 |
|----|------|------|
| 运行配置 | `config/arsi.yaml` | **api_key 必须为空**；密钥走环境 |
| 密钥 | `E:\ARSI\.env`（gitignore） | `ARSI_API_KEY=...` |
| 梦境超参 | `config/dream_rsi_params.yaml` | 官方 release 才改 |

**禁止**：把 key 写进 yaml / 提交 archive 运行态。

---

## 3. 周期任务

| 任务 | 频率 | 产物 |
|------|------|------|
| daemon tick | 300s | `archive/arsi_health.jsonl` |
| landscape + dry-run evolve | 每 3 tick | `archive/eval/harness_*` |
| GRPO 数据面导出 | 每 6 tick | `archive/eval/grpo_data_plane/` |
| dream_rsi_cycle | 按 dream_every_n | compare / manifest |

**dry-run 永不 apply core**；apply 需人工/门后。

---

## 4. 健康判据（探针口径）

| 检查 | 通过 |
|------|------|
| daemon 进程 | ≥1 个 `arsi_daemon` 且 working set > 20MB |
| lock | 存在且 pid 存活 |
| health 最新 tick | 时间戳 < 15 min |
| 铁律 | iron_laws 列表非空 |
| 测试 | `pytest tests` 全绿 |

---

## 5. 故障处置

| 症状 | 动作 |
|------|------|
| 双实例 / 抢锁 | `ops_stop` → 清 `archive/arsi.lock` → `ops_start` |
| health 停更 | 看 `daemon_err.log` → 重启 |
| pool 塌缩 | 查 `world_pool_snapshot.json`；persist 已防 shrink |
| LLM 401 | 核 `.env`；勿写回 yaml |
| 磁盘涨 | `ops_backup` 后清理 `archive/eval/compare_*.json` 旧文件 |

---

## 6. 投产检查清单

- [x] 密钥脱敏入库  
- [x] 全量测试绿（738+）  
- [x] 单实例锁  
- [x] 健康快照  
- [x] dry-run 默认  
- [ ] 备份 cron / 计划任务  
- [ ] 监控告警（磁盘 / 健康停更）  
- [ ] 对外 API 鉴权（若开 HTTP）

---

## 8. 冒烟与健康（已绿 2026-09-29）

```text
ops_smoke: ok=true (core/iron_laws/pool/SEVerA/epistemic/OPF/GRPO/no-secret)
ops_health: HEALTH_OK — pid 44684 ws 78MB · pool 42 · traces 719980
pytest: 738 passed
```

**投产判定（本机）**：daemon 稳定 · 密钥脱敏 · 契约 well-formed · 探针/备份脚本齐 · dry-run 默认。  
**未完（人工/运维）**：计划任务备份 · 磁盘/健康告警 · HTTP 鉴权。

- **ρ 不可改**：铁律 / sealed / 评分  
- 选择环禁语义；claim 负认识论  
- 外部依赖：AMD LLM API · hermes/mimo/synthex 本地数据
