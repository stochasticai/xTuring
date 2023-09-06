---
title: 🍽️ Setting up
description: Setup xTuring for contribution
sidebar_position: 1
---

Before we can start contributing to `xTuring`, we need to make sure we have all the neccessary code available to us in our working diretory and up-to-date. Moreover, to be able to test our changes, we need to do something more than just install the latest stable  version of the library from _pip_. 

To just get the latest version of developments on `xTuring`, we need to [install from source](#install-from-source). But in order to test our changes we did locally, we need to go a step ahead and do something called an [editable install](#editable-install). Let's dive into them right away!
## Install from source
Install `xTuring` directly from GitHub by running the following command on your cmd/terminal:
```bash
$ pip install git+https://github.com/stochasticai/xTuring
```
The above command will install the main version instead of the latest stable version of `xTuring`. This way of installing is a good way of being up-to-date with the latest development. There might be bugs which would be fixed in the main version but not yet rolled-out on pip. But at the same time this does not guarantee a stable version, this was of installation might break somewhere. We try our best to keep the main version free from any pitfalls and resolved of all the issues. If you run into a problem, please open up an [issue](https://github.com/stochasticai/xTuring/issues/new) and if possible, make a [contribution](https://github.com/stochasticai/xTuring/compare)!

Finally, you can test if `xTuring` has been properly installed by running the following commands on your cmd/terminal:
```bash
$ python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.

## Editable Install
The editable install is needed when you:
1. install directly from the source
2. wish to make a contribution to the `xTuring` library.

To do so, clone the library from _GitHub_ and install the necessary packages by running the following commands:
```bash
$ git clone https://github.com/stochasticai/xTuring.git
$ cd xTuring
$ pip install -e .
```
Now you will be able to test the changes you do to the library as Python will now look at `~/xTuring/` directory in addition to the normal installation of the library.

Next, in order to update your version, run the following command inside the `~/xTuring/` directory:
```bash
$ git pull
``` 
This will fast-forward your main version to the latest developments to the library, you can freely play around and use those.
