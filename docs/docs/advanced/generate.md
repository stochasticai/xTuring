---
title: 📚 Generate a dataset
description: Use self-instruction to generate a dataset
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- # Dataset generation -->

To generate a dataset we will make use of **engines** that consist of third party APIs. The below ones are the currently supported ones:

<Tabs>
  <TabItem value="openai" label="OpenAI">

  OpenAI api key can be obtained [here](https://beta.openai.com/account/api-keys)

  
```python
from xturing.model_apis.openai import ChatGPT, Davinci
engine = ChatGPT("your-api-key")
# or
engine = Davinci("your-api-key")
```
        
  </TabItem>
  
  <TabItem value="cohere" label="Cohere">

  ```python
  from xturing.model_apis.cohere import Medium
  engine = Medium("your-api-key")
  ```

  </TabItem>
  <TabItem value="ai21" label="AI21">

  ```python
  from xturing.model_apis.ai21 import J2Grande
  engine = J2Grande("your-api-key")
  ```

  </TabItem>
</Tabs>

## From no data

Even if we have no data, we can write a `.jsonl` file that contains the tasks/use cases we would like your model to perform well in. Continue reading to learn this file structure.

### Write your `tasks.jsonl`

Each line of this file needs to be a _JSON_ object with the following fields:

<!-- > ##### id (string, required)
> A unique identifier for the seed task. This can be any string that is unique within the set of seed tasks you are generating a dataset for.
>
> ##### name (string, required)
> A name for the seed task that describes what it is. This can be any string that helps you identify the task.
>
> ##### instruction (string, required)
> A natural language instruction or question that defines the task. This should be a clear and unambiguous description of what the task is asking the model to do.
>
>##### instances ([{input: string, output: string}, ...], required)
>A list of input-output pairs that provide examples of what the model should output for this task. Each input-output pair is an object with two fields: "input" and "output".
>
>##### is_classification (boolean, optional)
>A flag that indicates whether this is a classification task or not. If this flag is set to true, the output should be a single label (e.g. a category or class), otherwise the output can be any text. The default value is false. -->

| Name | Type | Desription |
| ---  | ---  | ---------- |
| __id__ |  string | A unique identifier for the seed task. This can be any string that is unique within the set of seed tasks you are generating a dataset for. |
| __name__ |  string | A name for the seed task that describes what it is. This can be any string that helps you identify the task. |
| __instruction__ |  string | A natural language instruction or question that defines the task. This should be a clear and unambiguous description of what the task is asking the model to do. |
| __instances__ |  List[Dict[str,str]] |  A list of input-output pairs that provide examples of what the model should output for this task. Each input-output pair is an object with two fields: __input__ and __output__. |
| __is_classification__ |  boolean | A flag that indicates whether this is a classification task or not. If this flag is set to true, the output should be a single label (e.g. a category or class), otherwise the output can be any text. The default value is false. |

Here's an example of a task in the above mentioned format: 

```json
{
    "id": "seed_task_0",
    "name": "addition",
    "instruction": "Add the two numbers together",
    "instances": [
        {
            "input": "2 + 2",
            "output": "4"
        },
        {
            "input": "3 + 7",
            "output": "10"
        }
    ],
    "is_classification": false
}
```

Here is how an sample `tasks.jsonl` file should like:

```json
{
  "id": "seed_task_0", 
  "name": "breakfast_suggestion", 
  "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?", 
  "instances": [{"input": "", "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."}], 
  "is_classification": false
}
{
  "id": "seed_task_1", 
  "name": "antonym_relation", 
  "instruction": "What is the relation between the given pairs?", 
  "instances": [{"input": "Night : Day :: Right : Left", "output": "The relation between the given pairs is that they are opposites."}], "is_classification": false
}
```

### Save the dataset

In order to use the dataset we just generated and not waste time again next we need it, we can simply save our instance like shown [here](/overview/quickstart/prepare#save-a-dataset).

### Example

Using `.generate_dataset()` method we can generate a dataset from a list of tasks/use cases. If the generation gets interrupted, since the results are being cached, we can resume the generation just by passing the same list of tasks. If we don't want to load the cached result, then we will just delete the created folder from our working directory.

```python
from xturing.datasets import InstructionDataset
from xturing.model_apis.openai import Davinci

## Load the required engine
engine = Davinci("your-api-key")

## Generate the dataset 
dataset = InstructionDataset.generate_dataset(path="./tasks.jsonl", engine=engine)

## Save the dataset instance
dataset.save('/path/to/directory')
```

Following parameters can be used to control the extent of generation:

| Name | Type| Default | Desription |
| ---  | --- | ----- | ---------- |
| __num_instructions_for_finetuning__ | int | 5  | The size of the generated dataset. If this number is much bigger than the number of lines in tasks.jsonl we can expect a more diverse dataset. Keep in mind that the __bigger the number__ you set, **more the credits** are going to be used from your engine. |
| __num_instructions__ | int | 10  | A cap on the size of the dataset, this can help to create a more diverse dataset. If you don't want to apply a cap, set this to the same value as *num_instructions_for_finetuning*.|

<!-- :::info
***generate_dataset()*** method accepts the following additional arguments:
```
...generate_dataset(
  num_instructions_for_finetuning=5,
  num_instructions=10
)
```

### num_instructions_for_finetuning
The size of the generated dataset. If this number is much bigger than the number of lines in tasks.jsonl we can expect a more diverse dataset. Keep in mind that the bigger the number you set **the more the credits** that are going to be used from your engine. The default value is 5.

### num_instructions
A cap on the size of the dataset, this can help to create a more diverse dataset. If you don't want to apply a cap, set this to the same value as *num_instructions_for_finetuning*. The default value is 10.

::: -->

## From custom data

We can also generate a dataset from our own files. 

<details>
  <summary>
  The files can be of one of the following formats:
  </summary>

  > .csv .doc .docx .eml .epub .gif .jpg .jpeg .json .html .htm .mp3 .msg .odt .ogg .pdf .png .pptx .rtf .tiff .tif .txt .wav .xlsx .xls

</details>

### Set up your environment
First, we need to make sure that all the necessary libraries are installed on our system. For this, we need to run the below commands:
<!-- Before going ahead we need to install some libraries that we help us with the text extraction. -->

<Tabs>
  <TabItem value="osx" label="OSX">


  This rely on you having [homebrew](http://brew.sh/) installed

  ```bash
  $ brew install caskroom/cask/brew-cask
  $ brew cask install xquartz
  $ brew install poppler antiword unrtf tesseract swig
  ```

  </TabItem>
  <TabItem value="ubuntu/debian" label="Ubuntu/Debian">

  ```bash
  $ apt-get update
  $ apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig
  ```

  </TabItem>
</Tabs>

### Prepare the files

Next, we just need to provide the directory path where our files are located. Files from sub-directories will also be discovered automatically.

### Save the dataset

In order to use the dataset we just generated and not waste time again next time we need it, we can simply save our instance like shown [here](/overview/quickstart/prepare#save-a-dataset).
<!-- ---
🚨 __CAUTION__:
- The file name is used as a context for the instruction generation. So, it is recommended to use meaningful names.
- Currently only **ChatGPT** engine is supported.
- Don't forget to [**save**](/overview/quickstart/prepare#save-a-dataset) the generated dataset.
--- -->
<!-- :::caution -->
<!-- ::: -->

### Example

```python
from xturing.datasets import InstructionDataset
from xturing.model_apis.openai import ChatGPT

# Load the required engine
engine = ChatGPT("your-api-key")

## Generate the dataset
dataset = InstructionDataset.generate_dataset_from_dir(path="/path/to/directory", engine=engine)

## Save the dataset instance
dataset.save("./my_generated_dataset")
```
<!-- We can print the `dataset` object to see the samples generated. -->

Following parameters can be used to customise data generation.

| Name | Type| Default | Desription |
| ---  | --- | ----- | ---------- |
| __use_self_instruct__ | bool | False|  When _True_ the dataset will be augmented with self-instructions (more samples, more diverse). In this case, you also have control hover the same parameters of *generate_dataset()* method: *num_instructions*, *num_instructions_for_finetuning*. |
| __chunk_size__ | int | 8000 |  The size of the chunk of text (in chars) that will be used to generate the instructions. We recommend values below 10000, but it depends on the model (engine) you are using. |
| __num_samples_per_chunk__ | int | 5 |  The number of samples that will be generated for each chunk. |


<!-- :::info
***generate_dataset_from_dir()*** method accepts the following additional arguments:
```
...generate_dataset_from_dir(
  use_self_instruct=False,
  chunk_size=8000,
  num_samples_per_chunk=5,
)
```

### use_self_instruct
When True the dataset will be augmented with self-instructions (more samples, more diverse). In this case, you also have control hover the same parameters of *generate_dataset()* method: *num_instructions*, *num_instructions_for_finetuning*. The default value is False.

### chunk_size
The size of the chunk of text (in chars) that will be used to generate the instructions. We recommend values below 10000, but it depends on the model (engine) you are using. The default value is 8000.

### num_samples_per_chunk
The number of samples that will be generated for each chunk. The default value is 5.
::: -->


<!-- --- -->

