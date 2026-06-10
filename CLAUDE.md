# CLAUDE.md - Aegis LLM working rules

## Scope

Aegis is the public trading core for risk, execution, bridges, backtests, competition, and model routing. Keep strategy implementation, credentials, account data, actors, host details, and private operational details out of this repository.

## Olympus Identity

Olympus work treats Aegis as an honest hypothesis falsifier and risk-discipline tool. Public documentation may describe the discipline and sanitized verdicts, but detailed strategy evidence stays in private incubating artifacts. Do not imply a verified alpha exists unless predeclared cost, benchmark, and OOS gates have passed.

## Review Integrity

- **Review must verify code.** A debrief is only an index for faster orientation. During review, independently verify claims with `git`, `rg`, `curl`, tests, and CI logs, and actively look for changes or gaps the debrief did not mention.
- **Debriefs must be independently verifiable.** Each material change must include file:line references, runnable verification commands, and test or CI results. Reviewers should be able to check the evidence without trusting conclusions.
- **Skipped work must be explicit.** Every debrief must include `Known Limits / Skipped / Trade-offs` and list unimplemented, simplified, deferred, or bypassed requirements with reasons. Do not use vague wording such as "optional", "can", "if needed", or "could" to hide skipped core requirements.
- **Debrief accuracy affects review score.** Misstating completion, omitting material gaps, or presenting skipped core requirements as done is a quality and integrity defect.
- **Read related reviews before executing.** Before any briefing, read relevant or recent files under `~/docs/reviews/`, incorporate their deficiencies and corrective suggestions, and either prioritize the correction or explicitly explain in the debrief why it was not done.
- **Gate claims require parameter disclosure.** Any debrief statement such as "gates unchanged", "safety not lowered", or "no risk relaxation" must enumerate every new or changed parameter that can affect gate strictness, execution eligibility, budget, notional, confirmation, exposure, or risk. For each such parameter, list its default, the standard/baseline value it differs from, and the reason for the difference. Do not use a technically true blanket statement to hide a newly introduced looser threshold. Reviewers must verify gate constants and new parameters in code rather than trusting summary claims.
- **Score-driven improvement.** Before executing an Olympus briefing, read the prior review 5-dimension score table and explicit deductions, align the work to those weak dimensions, and add a `Scoring Response` debrief section that states how PR handling, itemized disclosure, verification evidence, and honest negative verdicts addressed the prior review.

## Codex Execution Corrections

- Do not silently skip items merely because a briefing says "optional", "if", or "can". Evaluate them, decide whether they are core, and record done/not-done plus the reason in the debrief. If core status is unclear, stop and ask.
- Do not avoid judgment-heavy requirements such as write paths, retention, contradiction handling, safety boundaries, or edge-case policy. These are often the core of the task, not extras.
- Do not let green PRs pile up. After PR and CI are green, merge when authorized; otherwise state exactly who must review or merge and why it remains open.

## Debrief Standard

Every briefing debrief must follow `/home/gggqqy/docs/briefings/_DEBRIEF_TEMPLATE.md`, including `Verifiable Evidence` and `Known Limits / Skipped / Trade-offs`.

## Aegis Verification

- Use the repo Makefile, typically `make verify`, for canonical validation. It runs inside the Docker test container.
- Do not treat host-only dependency failures as canonical if the Makefile path works.
- Do not weaken paper defaults, confirmation gates, read-only semantics, health gates, or private-data boundaries.

## Olympus Research Discipline

- Start with an explicit thesis before implementing a candidate.
- Keep candidates interpretable; avoid hidden timing, leverage, or factor changes.
- Separate in-sample, out-of-sample, and walk-forward evidence.
- Use walk-forward or equivalent OOS checks before any robustness claim.
- Include all costs: fees, slippage, and funding or borrow costs when the instrument has perp, leverage, short, or borrow exposure. For spot long-only research, state funding/borrow as `N/A` explicitly.
- Compare against decision-relevant buy-and-hold or status-quo baselines, not only convenient strategy baselines.
- Report a standard metric block for every candidate and benchmark: max drawdown, Sharpe, Sortino, Calmar, positive-period win rate, OOS window win rate versus status quo where applicable, annualized turnover, and net cost.
- Do not connect research candidates to live trading APIs, order paths, or strategy registration without an explicit separate approval step.
