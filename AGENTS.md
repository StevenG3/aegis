# AGENTS.md — Deploy & Migrate Runbook for Hermes·TradingAgents·Aegis

> **Audience: an autonomous coding agent (Claude/Codex/etc.).** Read this top-to-bottom and you can
> deploy the whole stack on a fresh host, or migrate it to a new server/local machine, with minimal
> human input. All config is centralized into **one `secrets.env`**; in paper mode only **one LLM key**
> is mandatory. Commands are idempotent where possible. Stop and report on any STOP condition.

---

## 0. Operating rules for the agent
- Default to **paper mode**. Never set `MODE=live` / `LIVE_TRADING_ENABLED=true` unless the human explicitly asks.
- Never commit sensitive values. `.env` / `secrets.env` are gitignored; only `*.example` placeholders are tracked.
- All services bind `127.0.0.1`. Never bind `0.0.0.0` or publish to public internet.
- Verify each phase (healthz) before moving on. On failure, retry ≤3×, else STOP with logs.

---

## 1. Topology

| Repo | Clone URL | Path (suggested) | Visibility |
|---|---|---|---|
| aegis | `git@github.com:StevenG3/aegis.git` | `~/apps/aegis` | public |
| hermes-agent | `git@github.com:StevenG3/hermes-agent.git` | `~/apps/hermes-agent` | public |
| TradingAgents | `https://github.com/TauricResearch/TradingAgents.git` | `~/apps/tradingagents-official` | public |
| aegis-dashboard | `https://github.com/StevenG3/aegis-dashboard.git` | `~/aegis-dashboard` | **private** |
| atelier | `https://github.com/StevenG3/atelier.git` | `~/archives/atelier` | public |
| ops | `https://github.com/StevenG3/ops.git` | `~/ops` | **private** |

**Service ports (all `127.0.0.1`):** orchestrator 18081 · analysis-adapter 18085 · ibkr-bridge 18086 ·
exchange-bridge 18087 · backtest-service 18088 · TradingAgents bridge 18181 · dashboard 8910 ·
IBKR Gateway 4001/5900 · Hermes webhook 8644.

**Dependency order (bring-up):** TradingAgents bridge → Aegis stack → Hermes → Dashboard.

---

## 2. Prerequisites
```bash
docker --version && docker compose version && git --version
# container runtime data MUST live on SSD system disk (see §7); HDD is for cold data only
docker info | grep "Docker Root Dir"   # expect /var/lib/docker
```
- GitHub auth: SSH key or `gh auth login` (private repos need access to StevenG3/aegis-dashboard, StevenG3/ops).
- For Hermes Telegram entry: a bot token (optional — dashboard/CLI work without it).

---

## 3. Centralized config — ONE file

Create a single sensitive values file and distribute it. **This is the only file a human edits.**

```bash
mkdir -p ~/aegis-stack && cat > ~/aegis-stack/secrets.env <<'EOF'
# ===== MINIMAL (paper mode) — fill ONE LLM key, everything else optional =====
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
TA_DEFAULT_PROVIDER=deepseek            # match the key you filled

# ===== Optional: Telegram NL entry (Hermes) =====
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USERS=                 # CSV of tg ids; empty = deny all
AEGIS_ACTOR=tg_<YOUR_TELEGRAM_ID>
AEGIS_DEFAULT_ACTOR=tg_<YOUR_TELEGRAM_ID>

# ===== Safe defaults — DO NOT change for paper =====
MODE=paper
LIVE_TRADING_ENABLED=false
IBKR_LIVE_TRADING_ENABLED=false
IBKR_MODE=stub

# ===== Optional: LIVE only (leave <<unset>> for paper) =====
EXCHANGE_API_KEY=<<unset>>
EXCHANGE_API_SENSITIVE_VALUE=<<unset>>
OKX_API_KEY=<<unset>>
OKX_API_SENSITIVE_VALUE=<<unset>>
OKX_API_PASSPHRASE=<<unset>>
BYBIT_API_KEY=<<unset>>
BYBIT_API_SENSITIVE_VALUE=<<unset>>
CONFIRMATION_THRESHOLD_USDT=500
MAX_NOTIONAL_USDT=10000
EOF
chmod 600 ~/aegis-stack/secrets.env
```

**Distribute to each project (idempotent):**
```bash
# Aegis: start from example, then overlay shared sensitive values
cp -n ~/apps/aegis/deploy/.env.example ~/apps/aegis/deploy/.env
# append shared keys (compose only reads keys it knows; superset is fine)
grep -qxf ~/aegis-stack/secrets.env ~/apps/aegis/deploy/.env 2>/dev/null || \
  cat ~/aegis-stack/secrets.env >> ~/apps/aegis/deploy/.env

# TradingAgents: needs LLM keys
cp -n ~/apps/tradingagents-official/.env.example ~/apps/tradingagents-official/.env 2>/dev/null || true
cat ~/aegis-stack/secrets.env >> ~/apps/tradingagents-official/.env

# Hermes: env lives in its data dir, loaded at container start
mkdir -p ~/apps/data/hermes-official
cp ~/aegis-stack/secrets.env ~/apps/data/hermes-official/.env
```
> STOP if no LLM key is set (paper analysis cannot run).

---

## 4. Deploy runbook (in order, verify each)

```bash
# Phase 1 — TradingAgents bridge (:18181)
cd ~/apps/tradingagents-official && docker compose up -d --build
curl -s --max-time 5 localhost:18181/healthz || echo "TA bridge not ready"

# Phase 2 — Aegis stack (8 services)
cd ~/apps/aegis/deploy && docker compose up -d --build
for p in 18081 18085 18086 18087 18088; do
  printf "%s " $p; curl -s --max-time 5 localhost:$p/healthz; echo
done            # each should be {"status":"ok"}

# Phase 3 — Hermes gateway (loads ~/apps/data/hermes-official/.env at start)
cd ~/apps/hermes-agent/deploy && docker compose up -d
# deploy skills into the container data dir, then refresh skill manifest:
#   skills live at /opt/data/skills/domain/{aegis-gateway,atelier-gateway}
docker exec -u 10000 hermes rm -f /opt/data/.skills_prompt_snapshot.json 2>/dev/null || true
docker compose restart hermes

# Phase 4 — Dashboard (:8910, private repo)
cd ~/aegis-dashboard && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
DASHBOARD_PORT=8910 nohup python server.py >/tmp/dashboard.log 2>&1 &
curl -s --max-time 5 localhost:8910/ >/dev/null && echo "dashboard up"
```

---

## 5. Verification checklist
```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E "aegis|hermes|tradingagents"   # all healthy
curl -s localhost:18081/healthz                       # orchestrator ok
curl -s localhost:18088/strategies                    # backtest: lists ma_cross
curl -s localhost:18087/readyz                        # exchange readiness per venue
# Hermes NL routing (if Telegram/LLM configured):
docker exec -u 10000 hermes hermes chat -q "看我的余额"      # should route to aegis-gateway
docker exec -u 10000 hermes hermes chat -q "回测 BTCUSDT 20/50/200 均线"
```
PASS = all `{"status":"ok"}`, dashboard reachable, NL queries route to aegis-gateway.

---

## 6. Migration runbook (new server / local machine)

**6.1 Backup on old host**
```bash
# Aegis state volume (paper ledger, scorecards, etc.)
docker run --rm -v aegis_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/aegis_data.tgz -C /data .
# Hermes data dir (skills, .env, state)
tar czf hermes_data.tgz -C ~/apps/data/hermes-official .
# sensitive values
cp ~/aegis-stack/secrets.env ./secrets.env.bak
# repo list is in §1; code is on GitHub, no need to copy
```

**6.2 On new host**
```bash
# 1) prereqs (§2)  2) clone all repos (§1)  3) restore sensitive values to ~/aegis-stack/secrets.env
# 4) restore volumes BEFORE first up:
docker volume create aegis_data
docker run --rm -v aegis_data:/data -v "$PWD":/backup alpine \
  tar xzf /backup/aegis_data.tgz -C /data
mkdir -p ~/apps/data/hermes-official && tar xzf hermes_data.tgz -C ~/apps/data/hermes-official
# 5) distribute config (§3)  6) deploy (§4)  7) verify (§5)
```
> STOP if `aegis_data` volume restore fails — paper ledger/scorecards would be lost; do not start fresh silently.

---

## 7. Disk layout invariants (see ops/DISK_LAYOUT.md)
- **docker (`/var/lib/docker`) + containerd (`/var/lib/containerd`) BOTH on SSD system disk.** They are coupled; do not split across disks. Docker image layers physically live in containerd's content store.
- HDD `/mnt/blockstorage`: new projects (`projects/`, via `~/ops/new-project.sh`), backups, cold data.
- Control system-disk watermark: `docker builder prune -af` periodically + journald `SystemMaxUse=500M`.
- Diagnose "disk full": first check build-cache inflation (`docker system df` over-counts shared layers); use `sudo du -xhd1 /` for true usage (non-sudo du under-reports root-only dirs).
- `~/ops/check-disk.sh` + hourly cron watches `/`.

---

## 8. Safety invariants — MUST NOT break
1. Paper is default; live requires explicit `MODE=live` + `LIVE_TRADING_ENABLED=true` **and** a per-order confirmation token (fail-closed).
2. exchange-bridge & backtest-service are **read-only / simulation-only** — no `create_order`/`withdraw`.
3. All HTTP on `127.0.0.1`; dashboard never persists position/balance data.
4. No sensitive values/keys/host IPs in any repo. Private repos: aegis-dashboard, ops.
5. Risk-engine confirmation gate and live-order token checks must not be weakened.

---

## 9. Known gotchas
- **gh CLI 2.4.0**: no `gh pr checks --json`; `gh pr edit --label` fails (use `gh api repos/R/issues/N/labels`); `gh repo view --json visibility` unsupported (use `gh api repos/OWNER/REPO --jq .private`).
- **Hermes skill not routing**: SKILL.md `description` must be a routing trigger sentence (not a title); delete stale `/opt/data/.skills_prompt_snapshot.json` and restart; snapshot must be owned by hermes uid (10000), run `docker exec -u 10000`.
- **Container won't reach Aegis from Hermes**: use docker service DNS (`http://orchestrator:8080`) on the shared network, not `127.0.0.1`, inside containers.
- **rsync of docker/containerd data**: always `-aHAX` (`-H` for hardlinks; content store breaks without it).
- TradingAgents bridge entrypoint is `python -m hermes_bridge.api` on 18181; analysis-adapter reaches it via `TA_BRIDGE_URL`.
