# Contributing

Contributions are welcome! Here's how you can help.

## Reporting Issues

- Use the [GitHub Issues](https://github.com/nickirk/pytc/issues) page to report bugs or request features
- Include a minimal reproducible example when reporting bugs
- Describe the expected vs. actual behavior

## Submitting Pull Requests

1. Fork the repository and create a feature branch
2. Make your changes, following the existing code style
3. **Run the tests** before submitting:
   ```bash
   python -m unittest discover -v
   ```
4. Submit a pull request with a clear description of your changes

## Repository hygiene rules

These rules apply to all contributions.

**1. The repo holds durable assets only.**
Library code, tests, docs, and reproducible drivers. Run artifacts (JSON
results, log dumps, scratch files, one-off data) and provenance files
(cluster hostnames, job IDs, absolute paths) are never tracked. They live
in `artifacts/`, on the cluster, or in a results area outside the package.

**2. Behaviour is guaranteed by tests, not comments.**
If an invariant matters, write a test for it. Comments explain *why* (a
hidden constraint, a subtle invariant, a specific tradeoff); they do not
state *that* the invariant must hold — the test does that.

**3. No load-bearing one-time hacks.**
Manual env-var overrides (`PYTC_PANEL_BLK`, `PYTC_SOLVER_BLK`, …) are
documented expert escape hatches, never the default path. If a normal run
must hand-set a knob to avoid OOM or correctness failure, that is a bug in
the auto-sizing — fix the root cause.

**4. Caches are keyed on everything that changes the result.**
Any value that influences the cached computation must be part of the cache
key, with lookup and store using the same key. A cache that returns a stale
value when any input changes is a correctness bug.

**5. Review context is transient — do not fossilise it.**
Person-tagged comments ("Rick #26", "Woke #24", "task #N", "PR…") age
into noise. Convert each into a neutral, maintainable explanation or,
better, the test it was protecting. Run metadata (hashes, job IDs, commit
pins) belongs in commit messages or run logs, not in source files.

**6. Prefer the robust general fix over the targeted patch.**
When touching a hot path, reach for the fix that works for all callers
rather than one that papers over the symptom. Stage broad edits: report
findings and proposed changes before touching wide areas of the codebase.

## Code Style

- Follow the existing code conventions in the repository
- Use type hints where appropriate
- Add docstrings to new functions and classes

## Testing

```bash
# Canonical full suite
python -m unittest discover -v

# Coverage baseline (install with: python -m pip install -e '.[test]')
python -m coverage run -m unittest discover
python -m coverage report
```
