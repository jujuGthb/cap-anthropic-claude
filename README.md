# Anthropic Claude Capsule for NOVAVISION

A NOVAVISION capsule that integrates Anthropic's Claude API for image analysis and vision tasks.

## Features

- Image captioning (short & detailed)
- Object detection & classification
- Multi-label classification
- OCR (Optical Character Recognition)
- Visual Question Answering
- Structured data extraction
- Text prompting

## Requirements

- `anthropic` SDK
- `pydantic`
- `opencv-python`
- Anthropic API key

## Configuration

Each executor accepts the following common parameters:

| Parameter | Description |
|---|---|
| `api_key` | Anthropic API key |
| `model` | Claude model ID |
| `temperature` | Sampling temperature |
| `max_tokens` | Max tokens in response |
| `max_concurrent_requests` | Batch concurrency limit |

## Project Structure

```
src/
├── executors/    # One executor per task type
├── models/       # Pydantic I/O models
└── utils/        # Response builders
```
