---
title: ⚡️ FastAPI server
description: FastAPI inference server
sidebar_position: 3
---

# ⚡️ Running model inference with FastAPI Server

<!-- Once you have fine-tuned your model, you can run the inference using a FastAPI server. -->
After successfully fine-tuning your model, you can perform inference using a FastAPI server. The following steps guide you through launching and utilizing the API server for your fine-tuned model.

### 1. Launch API server from Command Line Interface (CLI)

To initiate the API server, execute the following command in your command line interface:

```sh
$ xturing api -m "/path/to/the/model"
```

:::info
Ensure that the model path you provide is a directory containing a valid xturing.json configuration file.
:::

### 2. Health check API

- ### Request

  - **URL** : http://localhost:{PORT}/health

  - **Method** : GET

- ### Response

  ```json
  {
    "success": true,
    "message": "API server is running"
  }
  ```

### 3. Inference API

- ### Request

  - **URL** : http://localhost:{PORT}/api

  - **Method** : POST

  - **Body** : The request body can contain the following properties:

    - **prompt**: Required, the prompt for text generation can be string or an array of Strings
    - **params**: Optional, Params for generation

    Here is an example for the request body:

    ```json
    {
      "prompt": ["What is JP Morgan?"],
      "params": {
        "penalty_alpha": 0.6,
        "top_k": 1.0,
        "top_p": 0.92,
        "do_sample": false,
        "max_new_tokens": 256
      }
    }
    ```

- ### Response

  ```json
  {
    "success": true,
    "response": ["JP Morgan is multinational investment bank and financial service headquartered in New York city."]
  }
  ```

By following these steps, you can effectively run your fine-tuned model for text generation through the FastAPI server, facilitating seamless inference with structured requests and responses.