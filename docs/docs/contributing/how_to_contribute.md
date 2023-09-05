---
title: How to contribute to xTuring?
description: How to contribute to xTuring?
sidebar_position: 2
---

<!-- # Contribute to xTuring -->

We are excited that you are interested in contributing to our open-source project. `xTuring` is an open-source library that is maintained by a community of developers and researchers. We welcome contributions of all kinds, including bug reports, feature requests, and code contributions.

## Ways to contribute
To start contributing to xTuring, we recommend that you familiarize yourself with our library by reading through our documentation and exploring the codebase. Once you are comfortable with the library, you can start contributing by:

1. **Fixing oustanding issues with existing code**
2. **Adding new models and features**
3. **Adding test cases**
4. **Maintaining code**
5. **Helping with documentation**

If you are skeptical where to start, here are some good [First Issues](https://github.com/stochasticai/xTuring/labels/good%20first%20issue) for you to go head on with.

### Fixing oustanding issues
If you come across some issue with the current code and have a solution to it, feel free to [contribute](https://github.com/stochasticai/xTuring/issues) and make a [Pull Request](https://github.com/stochasticai/xTuring/compare)! 


### Adding models and features
We are always looking to expand the capabilities of xTuring by adding more models and features. If you have expertise in a particular area of machine learning and would like to contribute a new model, we welcome your input. If you have an idea for a new feature, we encourage you to share your ideas with us by submitting an [issue](https://github.com/stochasticai/xTuring/issues)!

### Found some interesting test cases?
Adding tests is an essential part of maintaining and improving the quality of our codebase. By adding test cases, you can help ensure that xTuring is robust and reliable.

### Want to update the existing code?
Maintaining code is an important part of open-source development. By helping with code maintenance, you can ensure that xTuring is up-to-date, bug-free, and user-friendly.

### Wish to add to documentation?
As an open-source project, xTuring relies on documentation to help users understand how to use the library. If you have a talent for technical writing, we welcome your contributions to our documentation.


## Create a pull request
Before you make a PR, make sure to search through existing PRs and issues to make sure nobody else is already working on it. If unsure, it is best to open an issue to get some feedback.

In order to start and meet less hurdles on the way, it would be a good idea to be have _git_ proficiency to contribute to `xTuring`. If you get stuck somewhere, type _git --help_ in a shell and find your command!

You'll need Python 3.8 or above to contribute to __`xTuring`__. To start contributing, follow the below steps:

1. Fork the [repository](https://github.com/stochastciai/xTuring) by clicking on the [Fork](https://github.com/stochasticai/xTuring/fork) button on repository's GitHub page. This will create a copy of the main codebase under your GitHub account.

2. Clone your forked repository to your local machine, and add the base repository as a remote.

    ```bash
    $ git clone https://github.com/<YOUR_USERNAME>/xTuring.git
    $ cd xTuring
    $ git remote add upstream https://github.com/stochastic/xTuring.git
    ```

3. Create a new branch for your changes emerging from the `dev` branch.

    ```bash
    $ git checkout dev
    $ git checkout -b a-descriptive-name-for-your-changes
    ```
    🚨 **Do not** checkout from the _main_ branch or work on it.

4. Make sure you have pre-commit hooks installed and set up to ensure your code is properly formatted before being pushed to _git_. 

    ```bash
    $ pip install pre-commit
    $ pre-commit install
    $ pre-commit install --hook-type commit-msg
    ```

5. Set up a development environment by running the following command in a virtual environment:

    Here are the guides to setting up [virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/) and [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

    ```bash
    $ pip install -e .
    ```

    If `xTuring` is already installed in your virtual environment, remove it with _pip uninstall xturing_ before reinstlling it in editable mode with the _-e_ flag.

    To get a detailed version of installing the required libraries and `xTuring` for development, refer [here](/contributing/setting_up#editable-install).

6. Develop features on your branch
    As you work on your code, you should make sure the test suite passes. Run all the tests to make sure nothing breaks once your code is pushed to GitHub using the following command:
    ```bash
    $ pytest tests/
    ```

    Once you are satisified with your changes and all the tests pass, add changed files with _git add_ and record your changes with _git commit_:

    ```bash
    $ git add <modified files>
    $ git commit -m "<your commit message>"
    ```
    Make sure to write __good commit messages__ to clearly communicate the changes you made!

    Before pushing the code to GitHub and making a PR, ensure you have your copy of code up-to-date with the main repository. To do so, rebase your branch on _upstream/branch_:
    ```bash
    $ git fetch upstream
    $ git rebase upstream/dev
    ```

7. Push your changes to your forked repository

    ```bash
    $ git push -u origin a-descriptive-name-for-your-changes
    ```

8. Now, you can go your fork of the repository on GitHub and click on __[Pull Request](https://github.com/stochasticai/xTuring/compare)__ to open a pull request to the __dev__ branch of the original repository with a clear description of your changes and why they are needed. We will review your changes as soon as possible and provide feedback. Once your changes have been approved, they will be merged into the _dev_ branch.

9. Maintainers might request changes, it is fine if they do, it happens to our core contributors too! So everyone can see the changes in the pull request, work in your local branch and push the changes to your fork. They will automatically appear in the pull request.

We value the contributions of our community and are committed to creating an open and collaborative development environment. Thank you for considering contributing to xTuring!

### Sync a forked repository with upstream dec (the original xTuring repository)

To prevent triggering notifications and adding reference notes to each upstream pull request, adhere to these guidelines when making updates to the primary branch of a forked repository:

1. Whenever feasible, bypass synchronizing with the upstream repository by merging changes directly into the _dev_ branch of the forked repository. Evade the use of a separate branch and pull request for upstream synchronization.

2. In cases where a pull request is genuinely required, follow these steps once you've switched to your branch:

    ```bash
    $ git checkout -b your-branch-for-syncing
    $ git pull --squash --no-commit upstream dev
    $ git commit -m '<your message without GitHub references>'
    $ git push --set-upstream origin your-branch-for-syncing
    ```