# Building and Development

This document provides instructions for building the `llama-cpp-pydist` package from source and for updating its dependencies.

## Updating the `llama.cpp` Submodule

The `llama.cpp` C++ library is included as a git submodule in the `vendor/` directory. To update `vendor_llama_cpp_pydist/llama.cpp` to a new release tag, follow these steps:

1.  Navigate to the submodule directory:
    ```bash
    cd vendor_llama_cpp_pydist/llama.cpp
    ```

2.  Fetch the latest tags from the upstream repository:
    ```bash
    git fetch --tags origin
    ```

3.  Check out the desired new tag. Replace `new_tag_name` with the actual tag you want to use (e.g., `bxxxx`):
    ```bash
    git checkout new_tag_name
    ```

4.  Navigate back to the root directory of the main project:
    ```bash
    cd ../..
    ```

5.  Stage the changes to the submodule in the main project:
    ```bash
    git add vendor_llama_cpp_pydist/llama.cpp
    ```

6.  Commit the update in the main project:
    ```bash
    git commit -m "Update llama.cpp submodule to tag new_tag_name"
    ```

This will ensure that the main project now points to the updated version of the `llama.cpp` submodule.

## Building the Wheel

To build the Python wheel for this package, follow these steps:

1.  **Ensure `vendor` is a Package**:
    Make sure you have an empty `__init__.py` file in the `vendor` directory (`vendor/__init__.py`). This is necessary for the build system to recognize the `vendor` directory as a Python package and include its contents. If it doesn't exist, create it:
    ```bash
    touch vendor/__init__.py
    ```

2.  **Configure `setup.py` and `MANIFEST.in`**:
    *   In your `setup.py` file, ensure that `include_package_data=True` is set within the `setup()` function.
    *   Your `MANIFEST.in` file should include a line to recursively add all files from the `vendor_llama_cpp_pydist/llama.cpp` directory. For example:
        ```
        graft vendor_llama_cpp_pydist/llama.cpp
        ```
        You might also want to include other necessary files or prune unwanted ones using other `MANIFEST.in` commands.

3.  **Clean Previous Builds (Optional but Recommended)**:
    Before building, it's a good practice to clean any artifacts from previous builds:
    ```bash
    rm -rf dist build *.egg-info
    ```

4.  **Build the Wheel**:
    Run the following command from the root directory of the project (where `setup.py` is located):
    ```bash
    python3 setup.py bdist_wheel
    ```

### Offline UI Asset Bundling

`llama.cpp` now expects prebuilt UI assets in `vendor_llama_cpp_pydist/llama.cpp/tools/ui/dist` before it falls back to `npm` or network downloads. The package build automation stages those assets before creating the sdist and wheel so downstream builds can stay offline.

- `build_package.py` now runs the UI frontend build before `sdist` and `bdist_wheel`.
- The staged assets must include `index.html`, `loading.html`, and hashed bundle outputs under `tools/ui/dist/_app/immutable/`.
- The staged `dist/` directory is removed after packaging unless you pass `--preserve-staged-ui`.

Prerequisites for running `build_package.py`:

- `npm` must be available on the release/build machine.
- The vendored UI lockfile is used with `npm ci` when possible for reproducible staging.

To build release artifacts with bundled offline UI assets:

```bash
python3 build_package.py
```

To inspect the staged UI output after packaging:

```bash
python3 build_package.py --preserve-staged-ui
```

To verify the produced artifact is offline-safe, inspect the sdist or wheel contents and confirm these files are present:

- `vendor_llama_cpp_pydist/llama.cpp/tools/ui/dist/index.html`
- `vendor_llama_cpp_pydist/llama.cpp/tools/ui/dist/loading.html`
- `vendor_llama_cpp_pydist/llama.cpp/tools/ui/dist/_app/immutable/bundle*.js`
- `vendor_llama_cpp_pydist/llama.cpp/tools/ui/dist/_app/immutable/assets/bundle*.css`

This command will generate a `.whl` file in the `dist/` directory. This wheel file will contain the `vendor_llama_cpp_pydist/llama.cpp` directory and its contents, which are essential for the C++ backend.

You can then install the wheel using pip:
```bash
pip install dist/llama_cpp_pydist-*.whl
```
