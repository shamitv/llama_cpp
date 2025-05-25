# Llama CPP

This is a Python package for Llama CPP ( https://github.com/ggml-org/llama.cpp ).

## Installation

You can install the pre-built wheel from the releases page or build it from source.

```bash
pip install llama-cpp-pydist
```

## Usage

This section provides a basic overview of how to use the `llama_cpp_pydist` library.

### Deploying Windows Binaries

If you are on Windows, the package attempts to automatically deploy pre-compiled binaries. You can also manually trigger this process.

```python
from llama_cpp import deploy_windows_binary

# Specify the target directory for the binaries
# This is typically within your Python environment's site-packages
# or a custom location if you prefer.
target_dir = "./my_llama_cpp_binaries" 

if deploy_windows_binary(target_dir):
    print(f"Windows binaries deployed successfully to {target_dir}")
else:
    print(f"Failed to deploy Windows binaries or no binaries were found for your system.")

# Once deployed, you would typically add the directory containing llama.dll (or similar)
# to your system's PATH or ensure your application can find it.
# For example, if llama.dll is in target_dir/bin:
# import os
# os.environ["PATH"] += os.pathsep + os.path.join(target_dir, "bin")
```

For more detailed examples and advanced usage, please refer to the documentation of the underlying `llama.cpp` project and explore the examples provided there.

## Building and Development

For instructions on how to build the package from source, update the `llama.cpp` submodule, or other development-related tasks, please see [BUILDING.md](./BUILDING.md).
