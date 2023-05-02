# Contributing to xTuring

We welcome and appreciate contributions to xTuring! Whether it's a bug fix, a new feature, or simply a typo, every little bit helps.

## Getting Started

1. Fork the repository on GitHub
2. Clone your forked repository to your local machine

```bash
git clone https://github.com/<YOUR_USERNAME>/xturing.git
```

3. Create a new branch for your changes

```bash
git checkout -b <BRANCH_NAME>
```

4. Use pre-commit hooks to ensure your code is properly formatted

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

5. Make your changes and commit them

```bash
git add <FILES YOU ADDED/EDITED>
git commit -m "Commit message"
```

6. Push your changes to your forked repository

```bash
git push origin <BRANCH_NAME>
```

7. Create a pull request to the `dev` branch

## Pull Request Guidelines

Before submitting a pull request, please ensure the following:

1. Your changes are well-tested and do not break existing code
2. Your code adheres to the coding conventions used throughout the project
3. Your commits are properly formatted and have clear commit messages

## Bug Reports and Feature Requests

If you find a bug or have a feature request, please open an issue on GitHub. We'll do our best to address it as soon as
possible.

Thank you for contributing to xturing!
