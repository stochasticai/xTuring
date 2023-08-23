---
sidebar_position: 2
title: ⬇️ Installation
description: Your first time installing xTuring
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

You can install `xTuring` globally on your machine, but it is advised to install it inside a virtual environment. Before starting, make sure you have __Python 3.0+__ installed on your machine.

## Install via pip 
For this, ensure that you have _virtualenv_ package installed or _anaconda_ setup on your machine.

Start by creating a virtual environment in your working directory:

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

  </TabItem>
  <TabItem value="windows" label="Windows">

```bash
> venv\Scripts\Activate
```

  </TabItem>
</Tabs>

Once the virtual environment is activated, you can now install `xTuring` library by running the following command on your terminal:

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

  </TabItem>
  <TabItem value="windows" label="Windows">

```bash
> conda activate venv
```

  </TabItem>
</Tabs>

Once the conda environment is activated, you can now install `xTuring` library by running the following command on your terminal:


  </TabItem>
</Tabs>


```bash
$ pip install xTuring
``` 
This will install the latest version of xTuring available on pip.
Finally, you can test if `xTuring` has been properly installed by running the following commands on your terminal:
```bash
$ python
>>> from xturing.models import BaseModel
>>> model = BaseModel.create('opt')
>>> outputs = model.generate(texts=['Hi How are you?'])
```
Then print the outputs variable to see what the LLM generated based on the input prompt.
