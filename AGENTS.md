# Agent Runtime Guardrails

Use these defaults for this repository to avoid PowerShell stalls and approval churn.

1. Run shell commands with `-NoProfile` (or equivalent `login=false`) to avoid profile startup failures.
2. Execute shell commands sequentially only; do not batch or parallelize shell reads/writes.
3. When a task needs file writes, request one upfront escalated approval before making edits.
4. Keep discovery read-only first (`rg`, `Get-Content`), then perform a single write pass.
5. If a command is aborted/rejected, retry once sequentially; if it fails again, report the blocker immediately.
