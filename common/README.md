# Common - Shared Utilities

Shared library providing reusable utilities for other workspaces.

## Features

- **Images**: `common.images` for listing, resizing and iterating over images
- **Authentication**: cookie-based authentication for Flask web interfaces (`common.auth`, `common.serve`)
- **CLI**: `common.cli` for shared serve options and other command helpers
- **Progress**: `common.progress` for a shared progress bar factory
- **Configuration**: `common.config` and `common.paths` for project root and path management
- **Filesystem**: `common.fs` for atomic writes

## Usage

This is a library package with no CLI commands of its own. Other workspaces depend on it:

```python
from common import require_auth, run_app
```
