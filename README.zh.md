# Aegis

[English README](README.md)

Aegis 是 Olympus 的 public 通用交易内核仓库。它负责风控、编排、纸面执行、券商/交易所桥接、行情、回测、策略健康检查和研究证据的公共框架。

本仓库不是官方 TradingAgents 项目，也不应包含真实资金情况、账户标识、API 密钥、私有策略实现、私有策略验证结果、真实持仓、订单历史、代理端点、证书路径或主机私有运维日志。

## 当前定位

- 默认 paper-only，用于模拟执行、回测、研究和风控验证。
- live trading 路径必须显式开启，并经过风险检查和人工确认。
- Hermes 可以通过自然语言调用 Aegis 的分析、账户读取、回测和健康检查能力。
- TradingAgents 可以作为上游多智能体分析来源，但分析结果不能绕过 Aegis 风控。

## 运行

```bash
cd deploy
docker compose up --build
```

默认只有 orchestrator 暴露主机端口。生产或远程访问时，应使用 SSH tunnel 或私有网络，不要把交易服务直接暴露到公网。

## 常见用途

- 创建 paper order intent。
- 查看 paper positions 和 exposure。
- 读取 IBKR/交易所桥的只读账户状态。
- 运行回测、walk-forward、策略 healthcheck 和 competition ranking。
- 生成研究 evidence，并让人工审查是否继续推进。
- 通过 Hermes 执行 gateway doctor、backtest、position check、daily digest 等自然语言任务。

## Public 仓边界

public 仓可以包含：

- 通用服务代码。
- 占位符配置和 `.env.example`。
- 不含真实账户/资金/密钥的测试 fixture。
- 公共安全说明、架构文档和运行示例。
- 不泄露私有策略的通用研究框架。

public 仓不能包含：

- 真实账户号、余额、持仓、订单、盈亏或交易历史。
- Binance、OKX、Bybit、IBKR、OpenAI、Claude 等任何真实密钥或 token。
- 私有策略源码、私有策略验证结果、真实 competition 输出或 graduation evidence。
- 真实主机 IP、代理配置、证书路径或私有运维日志。

私有策略和验证产物应放在 private 仓库或被 `.gitignore` 忽略的本地目录中，例如 `aegis-strategies/` 或 `data/`。

## 安全原则

- 默认 paper，不默认实盘。
- live mode 必须多重开关和确认。
- 失败时关闭交易路径，而不是默认放行。
- 所有密钥只放本地 ignored env/config。
- Dashboard 和本地 API 应保持 `127.0.0.1` 绑定，通过 tunnel 访问。
