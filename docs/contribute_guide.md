# Contributing to NL2UniProt

Thank you for your interest in contributing to this project! This guide will help you get started with contributing to the project.

## Code Style and Quality

We use several tools to ensure code quality:

### Ruff & Pyright
We use Ruff for code linting and formatting, and Pyright for static type checking:

Before committing, you can run all code quality checks with:
```bash
ruff check --fix . && ruff format . && pyright
```
Make sure that you fix all ruff and pyright errors before committing. Otherwise, the commit will fail due to the pre-commit checks.

<!-- ## Testing

All code changes should include tests. We use pytest for testing:

1. Write tests for your changes in the `tests/` directory
2. Run the tests:
```bash
pytest
```

3. Ensure test coverage remains high
```bash
pytest --cov=leo_api tests/
``` -->

## Pull Request Process

1. Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them with clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: brief description of changes"
```

3. Before submitting:
   - Run all code quality checks
   - Run all tests (if you have written tests)
   - Update documentation if needed

4. Push your changes and create a Pull Request:
```bash
git push origin feature/your-feature-name
```

5. Create a Pull Request on GitHub:
   - Use a clear, descriptive title
   - Provide detailed description of changes
   - Reference any related issues
   - Fill out the PR template if provided

## PR Review Process

1. Maintainers (so far only me :() will review your PR
2. Address any requested changes
3. Once approved, maintainers will merge your PR

## Guidelines

- Keep changes focused and atomic
- Follow existing code style
- Include appropriate tests
- Update documentation
- Be respectful and collaborative

## Getting Help

If you need help:
1. Check existing documentation
2. Look through related issues
3. Open a new issue with questions
4. Tag maintainers for clarification

Thank you for contributing to NL2UniProt!