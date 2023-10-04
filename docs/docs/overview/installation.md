---
sidebar_position: 2
title: ⬇️ Installation
description: Your first time installing xTuring
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

We can install `xTuring` globally on our machine, but it is advised to install it inside a virtual environment. Before starting, we have to make sure we have __Python 3.0+__ installed on our machine.

## Install via pip 
For this, we have to ensure that we have _virtualenv_ package installed or _anaconda_ setup on our machine.

Start by creating a virtual environment in our working directory:

<Tabs>
  <TabItem value="venv" label="virtualenv">

```bash
$ virtualenv venv
```
Activate the virtual environment.

<Tabs>
  <TabItem value="unix" label="OSX/Linux">

```bash
$ source venv/bin/activate
```
Once the virtual environment is activated, we can now install `xTuring` library by running the following command on your terminal:

```bash
$ pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, we can test if `xTuring` has been properly installed by running the following commands on our terminal:
```bash
$ python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.


  </TabItem>
  <TabItem value="windows" label="Windows">

```bash
> venv\Scripts\Activate
```

Once the virtual environment is activated, we can now install `xTuring` library by running the following command on your terminal:

```bash
> pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, we can test if `xTuring` has been properly installed by running the following commands on our terminal:
```bash
> python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.


  </TabItem>
</Tabs>


  </TabItem>
  <TabItem value="conda" label="conda">

```bash
$ conda create -n venv
```
Activate the conda environment.
<Tabs>
  <TabItem value="unix" label="OSX/Linux">

```bash
$ conda activate venv
```


Once the conda environment is activated, we can now install `xTuring` library by running the following command on your terminal:

```bash
$ pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, we can test if `xTuring` has been properly installed by running the following commands on our terminal:
```bash
$ python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.


  </TabItem>
  <TabItem value="windows" label="Windows">

```bash
> conda activate venv
```


Once the conda environment is activated, we can now install `xTuring` library by running the following command on your terminal:

```bash
> pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, we can test if `xTuring` has been properly installed by running the following commands on our terminal:
```bash
> python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.


  </TabItem>
</Tabs>

  </TabItem>
</Tabs>


<!-- ```bash
$ pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, we can test if `xTuring` has been properly installed by running the following commands on our terminal:
```bash
$ python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt. -->
