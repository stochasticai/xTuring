---
title: How to contribute to xTuring?
description: How to contribute to xTuring?
sidebar_position: 1
---

# Contribute to xTuring

We are excited that you are interested in contributing to our open-source project. xTuring is an open-source library that is maintained by a community of developers and researchers. We welcome contributions of all kinds, including bug reports, feature requests, and code contributions.

### Ways to contribute
To start contributing to xTuring, we recommend that you familiarize yourself with our library by reading through our documentation and exploring the codebase. Once you are comfortable with the library, you can start contributing by:

1. **Adding more models and features**: We are always looking to expand the capabilities of xTuring by adding more models and features. If you have expertise in a particular area of machine learning and would like to contribute a new model, we welcome your input. If you have an idea for a new feature, we encourage you to share your ideas with us.
2. **Adding test cases**: Adding tests is an essential part of maintaining and improving the quality of our codebase. By adding test cases, you can help ensure that xTuring is robust and reliable.
3. **Maintaining code**: Maintaining code is an important part of open-source development. By helping with code maintenance, you can ensure that xTuring is up-to-date, bug-free, and user-friendly.
4. **Helping with documentation**: As an open-source project, xTuring relies on documentation to help users understand how to use the library. If you have a talent for technical writing, we welcome your contributions to our documentation.

### How to contribute
To contribute to xTuring, follow these steps:

1. Fork the repository on GitHub
2. Clone your forked repository to your local machine

```bash
git clone https://github.com/<YOUR_USERNAME>/xturing.git
```

3. Create a new branch for your changes emerging from the `dev` branch.

```bash
git checkout dev
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

7. Create a pull request to the `dev` branch with a clear description of your changes and why they are needed. We will review your changes as soon as possible and provide feedback. Once your changes have been approved, they will be merged into the dev branch.

We value the contributions of our community and are committed to creating an open and collaborative development environment. Thank you for considering contributing to xTuring!
