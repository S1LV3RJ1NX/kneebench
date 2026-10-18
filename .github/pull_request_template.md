## What and why

<!-- What does this change, and what problem does it solve? -->

## Checklist

- [ ] `uv run ruff format --check . && uv run ruff check . && uv run mypy` pass
- [ ] `uv run pytest -m "not live"` passes
- [ ] New or changed rules cite their source note and have ok, bad and edge-case tests
- [ ] User-facing text lives in the content layer, with no hardcoded models, GPUs or URLs
- [ ] No API keys or internal endpoint URLs in code, fixtures or screenshots
- [ ] CHANGELOG.md updated (if user-visible)
