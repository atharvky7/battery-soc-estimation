# Project Instructions for Codex

## Scope

The approved project scope is **AI-based initial State-of-Charge (SoC) estimation for lithium-ion batteries**. Continuous SoC tracking, charging-controller implementation, and hardware integration are outside the current deliverables. Do not expand scope without explicit approval from the user.

## Ticket discipline

- Work on only one approved ticket at a time. Future tickets in `docs/TASKS.md` are proposals until ChatGPT and the user approve them.
- Read the selected ticket's goal, priority, dependencies, allowed files, boundaries, and acceptance criteria before acting.
- Modify only files explicitly allowed by that ticket. Preserve existing information in files when asked to update them.
- Never implement unrelated features or refactor unrelated systems. If necessary work falls outside the approved ticket, record it as a follow-up issue.
- Prefer small, independently verifiable changes and avoid unnecessary dependencies.

## Preserve research artifacts and evidence

- Preserve datasets, model checkpoints, scalers, notebooks, saved experiment results, and historical reports. Do not overwrite, move, rename, or delete them unless the approved ticket explicitly allows that exact change.
- Never silently modify historical experimental results. Treat any approved result correction as a documented, traceable change and retain provenance.
- Never fabricate experimental results, successful tests, or validation evidence. Do not describe saved historical results as newly reproduced.
- Distinguish clearly between **implemented**, **tested**, and **verified** functionality. Presence of source code is not proof that it runs or is correct.
- Do not launch expensive training, simulations, or experiments without explicit approval.

## Repository and Git practices

- Use portable, repository-relative paths in new code and documentation; do not add machine-specific absolute paths.
- Never initialize Git, change Git configuration or remotes, stage or commit changes, or push to GitHub unless explicitly authorized for the active ticket.
- Do not install dependencies unless the active ticket explicitly authorizes it.

## Completion reports

At the end of each ticket, report:

1. Files changed.
2. Commands executed.
3. Tests or checks performed and their actual outcomes; say when none were run.
4. Known limitations and unresolved issues.
5. Follow-up tickets or decisions needed.

Do not claim a ticket is complete until its approved acceptance criteria are addressed or any remaining blocker is reported plainly.
