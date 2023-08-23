---
sidebar_position: 2
title: Installation
description: Your first time installing xTuring
---
<Reference: https://huggingface.co/docs/transformers/installation >

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Install via pip 
You can install `xTuring` globally on your machine, but it is advised to install it inside a virtual environment. Before starting, make sure you have __Python 3.0+__ installed on your machine and have _virtualenv_ package installed as well. 

Start by creating a virtual environment in your working directory:
```bash
$ virtualenv venv
```
Activate the virtual environment.

<Tabs>
  <TabItem value="unix" label="OSX/Linux">

```bash
$ source venv/bin/activate
```

  </TabItem>
  <TabItem value="windows" label="Windows">

```bash
> venv\Scripts\Activate
```

  </TabItem>
</Tabs>


Once the virtual environment is activated, you can now install `xTuring` library by running the following command on your terminal:
```bash
$ pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, you can test if `xTuring` has been properly installed by running the following commands on your cmd/terminal:
```bash
$ python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.

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
