## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for Abhinawap/financial-ml-regime-analysis, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Coding standards

Reviewers enforce `CODING_STANDARDS.md`. Mechanical rules live in `tests/test_architecture.py`, ruff and vulture (`pyproject.toml`).

### Spec

`docs/spec.md` is a copy of #1. Behaviour the spec doesn't define goes on #1 as a dated amendment comment, copied word for word into `docs/spec.md` before merge.
