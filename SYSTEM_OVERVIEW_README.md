# Hermes · TradingAgents · Aegis — System Overview / 系统总览

> An AI-operated trading stack: natural language in → multi-agent market analysis → risk-gated order execution for US/HK stocks and crypto, with a private dashboard and an autonomous Claude×Codex development loop.
>
> 一套 AI 驱动的交易系统:自然语言输入 → 多智能体行情分析 → 经风控闸门的下单执行(美股/港股/加密),配私有可视化看板,以及 Claude×Codex 自主开发协作环。

---

## 1. What is this / 这是什么

**EN.** Three layers cooperate: **Hermes** is the natural-language gateway (Telegram + local dashboard), **TradingAgents** is the multi-agent analysis brain, and **Aegis** is the trading core (8 microservices for risk, execution, broker/exchange bridges, market data, backtesting). Everything runs in Docker on one host and is **paper-trading by default** — no real money moves until you explicitly enable live mode.

**中文.** 三层协作:**Hermes** 是自然语言网关(Telegram + 本地看板),**TradingAgents** 是多智能体分析大脑,**Aegis** 是交易内核(8 个微服务,负责风控、下单、券商/交易所桥接、行情、回测)。全部以 Docker 跑在单台主机上,**默认纸面交易(paper)** —— 在你显式开启实盘前,不会动用任何真实资金。

---

## 2. Repositories / 代码仓库

| Project | GitHub | Visibility | Role / 作用 |
|---|---|---|---|
| **Aegis** | https://github.com/StevenG3/aegis | public | Trading core, 8 microservices / 交易内核,8 微服务 |
| **Hermes Agent** | https://github.com/StevenG3/hermes-agent | public | NL gateway + skills routing / 自然语言网关 + 技能路由 |
| **TradingAgents** | https://github.com/TauricResearch/TradingAgents | public (upstream) | Multi-agent analysis brain / 多智能体分析大脑 |
| **Aegis Dashboard** | https://github.com/StevenG3/aegis-dashboard | private | Local positions/PnL/backtest UI / 本地持仓·盈亏·回测看板 |
| **Atelier** | https://github.com/StevenG3/atelier | public | Claude×Codex dev collaboration / 开发协作框架 |
| **Ops** | https://github.com/StevenG3/ops | private | Disk layout & host runbooks / 磁盘布局与运维脚本 |

---

## 3. Architecture / 整体架构

```
                    ┌─────────────────────────────────────────────┐
   Telegram /       │                 HERMES (gateway)            │
   Browser  ───────▶│  NL → skills router                          │
   自然语言          │   • aegis-gateway skill  → Aegis             │
                    │   • atelier-gateway skill→ GitHub kanban     │
                    └───────────────┬─────────────────────────────┘
                                    │ HTTP (127.0.0.1, internal net)
                                    ▼
        ┌──────────────────────── AEGIS (trading core) ───────────────────────┐
        │  orchestrator:18081  ──▶ risk-engine ──▶ execution-service           │
        │        │                  (limits,           │   (paper/live,         │
        │        │                   confirm gate)      │    live needs token)  │
        │        ├──▶ analysis-adapter:18085 ──▶ TradingAgents bridge :18181   │
        │        ├──▶ market-data ──▶ ibkr-bridge:18086 ──▶ IBKR Gateway       │
        │        ├──▶ exchange-bridge:18087  (ccxt: OKX/Binance/Bybit, read)   │
        │        └──▶ backtest-service:18088 (backtesting.py)                  │
        └──────────────────────────────┬───────────────────────────────────────┘
                                        │ read-only proxy
                                        ▼
                          Aegis Dashboard :8910 (127.0.0.1 only)
                          持仓 / 盈亏 / 对账 / 交易所余额 / 回测可视化

   Dev loop (out-of-band):  Claude (review+spec) ⇄ GitHub ⇄ Codex (implement)  — via Atelier
```

**Data flow / 数据流:** User sends NL → Hermes routes to aegis-gateway skill → Aegis orchestrator coordinates analysis (TradingAgents), risk checks, and execution (IBKR for stocks / exchanges for crypto). Hermes also receives fill webhooks. The dashboard reads Aegis read-only for visualization.
用户发自然语言 → Hermes 路由到 aegis-gateway 技能 → Aegis orchestrator 协调分析(TradingAgents)、风控、下单(美股走 IBKR / 加密走交易所)。Hermes 还接收成交回调。看板只读 Aegis 做可视化。

---

## 4. Ports / 端口

| Service | Host port | 说明 |
|---|---|---|
| Aegis orchestrator | 127.0.0.1:18081 | Main API / 主入口 |
| analysis-adapter | 127.0.0.1:18085 | → TradingAgents |
| ibkr-bridge | 127.0.0.1:18086 | Stocks via IBKR / 美股港股 |
| exchange-bridge | 127.0.0.1:18087 | Crypto balances (read) / 加密余额(只读) |
| backtest-service | 127.0.0.1:18088 | Backtests / 回测 |
| TradingAgents bridge | 127.0.0.1:18181 | Analysis API / 分析接口 |
| Aegis Dashboard | 127.0.0.1:8910 | Local UI / 本地看板 |
| IBKR Gateway | 127.0.0.1:4001, 5900 | Broker gateway / 券商网关 |

> All services bind to `127.0.0.1` only. To view remotely, use an SSH tunnel — never expose to the public internet.
> 所有服务只监听 `127.0.0.1`。远程查看请用 SSH 隧道,切勿暴露公网。

---

## 5. Quick start (paper mode) / 快速开始(纸面模式)

**EN.** Paper mode needs almost no sensitive values — only **one LLM key** for analysis. Live trading is off by default.

**中文.** 纸面模式几乎不需要任何密钥 —— 只要**一个大模型 key** 供分析用。实盘默认关闭。

```bash
# 1. Prereqs / 前置: docker, docker compose, git
# 2. Clone core repos / 克隆核心仓库
git clone https://github.com/StevenG3/aegis.git
git clone https://github.com/StevenG3/hermes-agent.git
git clone https://github.com/TauricResearch/TradingAgents.git tradingagents-official

# 3. Minimal config / 最小配置 (see §6) — set ONE LLM key
cp aegis/deploy/.env.example aegis/deploy/.env          # defaults are safe (paper)

# 4. Bring up Aegis (paper) / 启动 Aegis
cd aegis/deploy && docker compose up -d --build

# 5. Verify / 验证
curl -s localhost:18081/healthz          # {"status":"ok"}
```

> For the full, machine-executable deployment (TradingAgents + Hermes + dashboard, with verification), see **[AGENTS.md](./AGENTS.md)**.
> 完整的、可由机器执行的部署(含 TradingAgents + Hermes + 看板 + 验证),见 **[AGENTS.md](./AGENTS.md)**。

---

## 6. Configuration — minimal by design / 配置(最小化设计)

**EN.** The system is built so **almost everything has a safe default**. You only fill in what you actually use.

**中文.** 系统设计为**几乎一切都有安全默认值**。你只需填你真正要用的部分。

| Variable / 变量 | Required? / 必填 | Default / 默认 | Purpose / 用途 |
|---|---|---|---|
| `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` *or* `DEEPSEEK_API_KEY` | **Yes (1 of)** / 三选一 | — | LLM for analysis / 分析用大模型 |
| `TELEGRAM_BOT_TOKEN` | Only for Telegram / 仅用 Telegram 时 | — | Hermes NL entry / 自然语言入口 |
| `TELEGRAM_ALLOWED_USERS` | With Telegram / 配合上条 | empty = deny all / 空=全拒 | Access control / 访问控制 |
| `MODE` | No | `paper` | `paper` or `live` |
| `LIVE_TRADING_ENABLED` | No | `false` | Master live switch / 实盘总开关 |
| `EXCHANGE_API_KEY and exchange sensitive value`, `OKX_*`, `BYBIT_*` | Only for live crypto / 仅实盘加密 | `<<unset>>` | Exchange read/trade keys |
| `IBKR_*` | Only for live stocks / 仅实盘美股 | stub | IBKR Gateway connection |
| `CONFIRMATION_THRESHOLD_USDT` | No | `500` | Orders above need confirm / 超额需确认 |

**Live trading checklist / 实盘开启清单:** set `MODE=live` + `LIVE_TRADING_ENABLED=true`, add broker/exchange keys, and every live order still requires a confirmation token (fail-closed). / 设 `MODE=live` 且 `LIVE_TRADING_ENABLED=true`,填券商/交易所 key;**每笔实盘单仍需确认令牌**(fail-closed 安全设计)。

---

## 7. Safety model / 安全模型

- **Paper by default / 默认纸面** — no real orders until live is explicitly enabled. / 实盘显式开启前不下真实单。
- **Confirmation gate / 确认闸门** — live orders fail-closed without a token; large notionals always need confirmation. / 实盘单无令牌即拒;大额必确认。
- **Local-only surfaces / 仅本地** — all HTTP on `127.0.0.1`; dashboard never persists data; access remotely via SSH tunnel. / 全部 `127.0.0.1`;看板不落盘;远程用 SSH 隧道。
- **Sensitive values never committed / 密钥不入库** — `.env` is gitignored; only `.env.example` placeholders ship. / `.env` 被忽略,仓库只含占位符。
- **Read-only bridges / 只读桥** — exchange-bridge & backtest never place orders. / 余额查询与回测绝不下单。

---

## 8. Host & disk / 主机与磁盘

**EN.** Container runtimes (docker + containerd) live on the **SSD system disk**; the secondary HDD (`/mnt/blockstorage`) holds new projects, backups and cold data. Disk watermark is monitored hourly. See `ops/DISK_LAYOUT.md`.

**中文.** 容器运行时(docker + containerd)放在 **SSD 系统盘**;副 HDD(`/mnt/blockstorage`)放新工程、备份与冷数据。磁盘水线每小时监控。详见 `ops/DISK_LAYOUT.md`。

---

## 9. Development model — Atelier / 开发模式

**EN.** Code evolves through a **Claude × Codex** loop: Claude reviews and writes specs (`CODEX_*.md` prompts); Codex implements, tests, and opens PRs. The two never talk directly — **GitHub Issues/PRs are the only channel**. See the [Atelier](https://github.com/StevenG3/atelier) repo.

**中文.** 代码通过 **Claude × Codex** 协作环演进:Claude 审查并写规格(`CODEX_*.md` 提示词),Codex 实现、测试、开 PR。两者从不直接对话 —— **GitHub Issues/PR 是唯一通道**。详见 [Atelier](https://github.com/StevenG3/atelier)。

---

## 10. Remote access / 远程访问

```bash
# View the dashboard from your laptop / 从本地查看看板
ssh -L 8910:127.0.0.1:8910 <user>@<SERVER_IP>
# then open / 然后打开  http://127.0.0.1:8910
```

> Replace `<user>` and `<SERVER_IP>` with your own. Never bind services to `0.0.0.0`.
> 把 `<user>`、`<SERVER_IP>` 换成你自己的。切勿把服务绑到 `0.0.0.0`。
