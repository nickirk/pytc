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
