---
title: 🤔 FAQs
description: Some common issues one might have 
sidebar_position: 9
---

**How to fine-tune a LLM?**
    
You can refer [here](/overview/quickstart/finetune_guide) for a tutorial on how to fine-tune any model of your choice. The list of all the supported model can be found on [this page](/overview/supported_models).

**How to load a model not there in the [supported models](/overview/supported_models)?**

If you cannot find the model you want to use in the support models' list, then you can refer to [this guide](/advanced/anymodel) on how to load any other model of your choice with ease.

**How to use an existing dataset for instruction fine-tuning?**

A tutorial on this can be found [here](/overview/quickstart/data_usage), where you will see how to load Alpaca Dataset and prepare it in instruction fine-tuning format.

**How to setup xTuring to start contributing?**

To setup the environment ready to contribute to xTuring, you need to do an _editable install_ of the source on your machine. The steps for the same are mentioned [here](/contributing/setting_up#editable-install).

**Which all fine-tuning memory-efficient techniques are supported by xTuring and how to use them with models not on the supported models' list?**

Other than normal LLM fine-tuning, xTuring supports:
- Low-Rank Adaption (LoRA), 
- 8-bit precision, 
- LoRA with 8-bit precision and 
- LoRA with 4bit precision 

techniques optimized for low-resource fine-tuning. To use any model not mentioned on the [supported models' list](/overview/suppported_models), you can refer [this page](/advanced/anymodel) to fine-tune a model of your choice with the preferred technique.