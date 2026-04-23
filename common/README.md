# Common - Shared Utilities

Shared library providing reusable utilities for other workspaces.

## Features

- **CLIP Model Loading**: `get_model()`, `get_processor()`, `get_device()` for loading CLIP models with automatic device selection (CUDA/CPU)
- **Data Loading**: `get_dataloader()` for batched image loading with PyTorch DataLoaders
- **Authentication**: Cookie-based authentication for Flask web interfaces
- **Configuration**: Project root path management and environment variable loading

## Usage

This is a library package with no CLI commands. Other workspaces depend on it:

```python
from common import get_model, get_processor, get_device, get_dataloader
```
