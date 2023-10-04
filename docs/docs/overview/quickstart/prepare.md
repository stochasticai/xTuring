---
title: 💽 Prepare and save dataset
description: Use self-instruction to generate a dataset
sidebar_position: 2
---

<!-- ## Prepare Instruction dataset -->


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- # Using datasets -->

We provide several type of datasets to use with your data. Depending on how you want to train and use your model, you can choose:
- [**InstructionDataset**](#instructiondataset) - You want the model to generate text based on an instruction/task.
- [**TextDataset**](#textdataset) - You want the model to complete your text.

## InstructionDataset

Here is how you can create this type of dataset:

<Tabs>
<TabItem value="dictionary" label="Dictionary">

From a python dictionary with the following keys:

- **instruction** : List of strings representing the instructions/tasks.
- **text** : List of strings representing the input text.
- **target** : List of strings representing the target text.


```python
from xturing.datasets.instruction_dataset import InstructionDataset

dataset = InstructionDataset({
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
    "instruction": ["first instruction", "second instruction"]
})
```

</TabItem>
<TabItem value="folder" label="Folder">

From a saved location:


```python
from xturing.datasets.instruction_dataset import InstructionDataset

dataset = InstructionDataset('path/to/saved/location')
```

</TabItem>
</Tabs>

## TextDataset

Here is how you can create this type of dataset:

<Tabs>
<TabItem value="dictionary" label="Dictionary">

From a python dictionary with the following keys:

- **text** : List of strings representing the input text.
- **target** : List of strings representing the target text.


```python
from xturing.datasets.text_dataset import TextDataset

dataset = TextDataset({
    "text": ["first text", "second text"],
    "target": ["first text", "second text"]
})
```

</TabItem>
<TabItem value="folder" label="Folder">

From a saved location:


```python
from xturing.datasets.text_dataset import TextDataset

dataset = TextDataset('path/to/saved/location')
```

</TabItem>
</Tabs>

## Save a dataset

You can save a dataset to a folder using the `save` method:

```python
from xturing.datasets import ...
dataset = ...

dataset.save('path/to/a/directory')
```

