---
title: ⚡️ FastAPI server
description: FastAPI inference server
sidebar_position: 3
---

# ⚡️ Running model inference with FastAPI Server

<!-- Once you have fine-tuned your model, you can run the inference using a FastAPI server. -->
After successfully fine-tuning your model, you can perform inference using a FastAPI server. The server exposes:

1. Legacy route: `/api`
2. OpenAI-compatible routes: `/v1/models` and `/v1/chat/completions`

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

### 3. Legacy inference API

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

### 4. OpenAI-compatible API

#### List models

- **URL**: `http://localhost:{PORT}/v1/models`
- **Method**: `GET`

Response example:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3_0_6b_lora",
      "object": "model"
    }
  ]
}
```

#### Chat completions

- **URL**: `http://localhost:{PORT}/v1/chat/completions`
- **Method**: `POST`

Request example:

```json
{
  "model": "qwen3_0_6b_lora",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me 3 steps to start fine-tuning."}
  ],
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 128
}
```

Response example:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen3_0_6b_lora",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Step 1..."},
      "finish_reason": "stop"
    }
  ]
}
```

By following these steps, you can run legacy and OpenAI-compatible inference from the same xTuring API server.
