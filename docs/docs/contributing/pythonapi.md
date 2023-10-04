---
title: 🐍 Python API
description: Developers' guide for API
sidebar_position: 4
---

<!-- ## Python API -->

This section includes the API documentation from the Finetuner codebase, as extracted from the [docstrings](https://peps.python.org/pep-0257/) in the code.

## `BaseModel`
<!-- <table style={{ width: "100%" }}>
    <tr>
        <td> `BaseModel.load` </td>
        <td>Load a model from your local machine or xTuring Hub.</td>
    </tr>
</table> -->
[`BaseModel.load(weights_dir_or_model_name)`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/models/base.py#L15)
> Load a model from your local machine or xTuring Hub.
>
> **Parameters**:
> - **weights_dir_or_model_name** *(str)*: Path to a local model to be load or a model from `xTuring` Hub. 

<!-- | Method | Description |
| --------- | ----------- |
|`BaseModel.load` | Load a model from your local machine or xTuring Hub. | -->

## `CausalModel`

[`model.finetune(dataset, logger = True)`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/models/causal.py#L116)
> Fine-tune the in-memory model on the desired dataset.
> 
> **Parameters**:
> - **dataset** *(Union[TextDataset, InstructionDataset])*: The object of either of the 2 dataset classes specified in the library. If not passed, will throw an error.
> - **logger** *(Union[Logger, Iterable[Logger], bool])*: If you want to log the progress in the default logger, pass nothing explicitly. Else, you can pass your own logger.

[`model.generate(texts = None ,dataset = None, batch_size = 1)`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/models/causal.py#L158)
> Use the in-memory model to generate outputs by passing either a `dataset` as an argument or `texts` as an argument which would be a list of strings.
> 
> **Parameters**:
> - **texts** *(Optional[Union[List[str], str]])*: Can be a single string or a list of strings on which you want to test your in-memory model.
> - **dataset** *(Optional[Union[TextDataset, InstructionDataset]])*: The object of either of the 2 dataset classes specified in the library.
> - **batch_size** *(Optional[int])*: For faster processing given your machine constraints, you can configure the batch size of the model. Higher the batch size, more the parallel compute, faster you will get your result.

[`model.evaluate(dataset, batch_size = 1)`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/models/causal.py#L312)
> Evaluate the in-memory model.
> 
> **Parameters**:
> - **dataset** *(Optional[Union[TextDataset, InstructionDataset]])*: The object of either of the 2 dataset classes specified in the library.
> - **batch_size** *(Optional[int])*: For faster processing given your machine constraints, you can configure the batch size of the model. Higher the batch size, more the parallel compute, faster you will get your result.

[`model.save(path)`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/models/causal.py#L212)
> Save your in-memory model.
>
> **Parameters**:
> - **path** *(Union[str, Path])*: The path to the directory where you want to save your in-memory model. Can either be a string or a `Path` object (class found in _pathlib_).


## `InstructionDataset`

[`dataset.from_jsonl`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/datasets/instruction_dataset.py#L80)
> Get an instruction data from a `.jsonl` file where each line is a json object with keys _text_, _instruction_ and _target_.
>
> **Parameters**:
> - **path** *(Path)*: the path to the _.jsonl_ file. Should be an object of the class `Path` from the _pathlib_.

[`InstructionDataset.generate_dataset`](https://github.com/stochasticai/xTuring/blob/9b98c68af8391c7a0f48a141178b70a1a8e47c06/src/xturing/datasets/instruction_dataset.py#L127)
> Generate your custom dataset given the HuggingFace engine.
>
> **Parameters**:
> - **path** *(str)*: a string of the path where you want to save the generated dataset.
> - **engine** *(TextGenerationAPI)*: should be an object of one of the classes mentioned in the [*model_apis*](https://github.com/stochasticai/xTuring/tree/main/src/xturing/model_apis) directory. 
> - **num_instructions** *(Optinoal[int])*: a cap on the size of sample set to be generated. Helps you create a more diverse dataset.
> - **num_instructions_for_finetuning** *(Optinoal[int])*: size of the sample set to be generated. Uses up the credits from your account. <u>_Use this number very carefully._</u>
<!-- > - **num_prompt_instructions** *(Optinoal[int])*:  -->

<!-- | Method | Description |
| --------- | ----------- |
| `model.finetune` |  Fine-tune the in-memory model on the desired dataset by passing the argument `dataset` in the function call.   |
| `model.generate` |  Use the in-memory model to generate outputs by passing either a `dataset` as an argument or `texts` as an argument which would be a list of strings.    |
| `model.evaluate` |  Evaluate the in-memory model    |
 -->

<!-- ## `CausalEngine.__init__`
| Parameter | Description |
| --------- | ----------- |
| `CausalEngine.model_name` | Print name of the pre-trained LLM. |
| `CausalEngine.model` | Print the LLM class loaded from HuggingFace Hub. |
| `CausalEngine.tokenizer` | Print the tokenizer class being used. |
| `CausalEngine.load_8bit` | Whether the model is loaded in INT8 preicision. |
| `CausalEngine.trust_remote_code` | To download the model with weights onto your machine and then run it. |

## `CausalLoraEngine.__init__`
| Parameter | Description |
| --------- | ----------- |
| `CausalLoraEngine.model_name` | Print name of the pre-trained LLM. |
| `CausalLoraEngine.model` | Print the LLM class loaded from HuggingFace Hub. |
| `CausalLoraEngine.tokenizer` | Print the tokenizer class being used. |
| `CausalLoraEngine.load_8bit` | Whether the model is loaded in INT8 preicision. |
| `CausalLoraEngine.trust_remote_code` | To download the model with weights onto your machine and then run it. |
| `CausalLoraEngine.target_modules` | The layers in the LLMs which where Low-Rank Adaption LoRA will be applied. |


## classs `CausalModel.__init__`
(
    engine: str,
    weights_path: Optional[str] = None,
    model_name: Optional[str] = None,
    target_modules: Optional[List[str]] = None,
    **kwargs
)
| Parameter | Description |
| --------- | ----------- |
| `CausalModel.model_name` | Print name of the pre-trained LLM. |
| `CausalModel.engine` | Print the engine class loaded for the xTuring model class. |
| `CausalModel.tokenizer` | Print the tokenizer class being used. |
| `CausalModel.load_8bit` | Whether the model is loaded in INT8 preicision. |
| `CausalModel.trust_remote_code` | To download the model with weights onto your machine and then run it. |
| `CausalModel.target_modules` | The layers in the LLMs which where Low-Rank Adaption LoRA will be applied. |
engine
model_name
finetuning_args
generation_args -->