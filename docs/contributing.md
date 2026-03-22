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
   python -m unittest discover pytc/test
   python -m unittest discover pytc/ansatz/test
   python -m unittest discover pytc/vmc/test
   python -m unittest discover pytc/jastrow/test
   ```
4. Submit a pull request with a clear description of your changes

## Code Style

- Follow the existing code conventions in the repository
- Use type hints where appropriate
- Add docstrings to new functions and classes

## Testing

```bash
# Core tests
python -m unittest discover pytc/test

# Submodule tests
python -m unittest discover pytc/ansatz/test
python -m unittest discover pytc/vmc/test
python -m unittest discover pytc/jastrow/test
python -m unittest discover pytc/solver/test

# Legacy tests (NumPy implementation)
python -m unittest discover pytc/legacy/test
```
