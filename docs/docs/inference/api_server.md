---
title: FastAPI server
description: FastAPI inference server
sidebar_position: 3
---

Once you have fine-tuned your model, you can run the inference using a FastAPI server.

### 1. Launch API server from CLI

```sh
xturing api -m "/path/to/the/model"
```

:::info
Model path should be a directory containing a valid `xturing.json` config file.
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
