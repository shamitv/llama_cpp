\
# Build Notes for `llama_cpp_pydist`

This document outlines the process and rationale for building the `llama_cpp_pydist` Python package, particularly concerning the inclusion and management of the `vendor/llama.cpp` C++ submodule.

## Goal

The primary goal is to create a Python wheel (`.whl`) that:
1.  Includes the necessary source code from the `vendor/llama.cpp` submodule.
2.  Excludes unnecessary large directories from `vendor/llama.cpp` (like `models/` and the original `examples/`) to keep the package size reasonable.
3.  Ensures that the packaged C++ code can be successfully built when the wheel is installed by a user (e.g., via `pip install`).

## Challenges and Solutions

### 1. Including the Submodule

-   **Challenge**: Standard Python packaging tools might not automatically include files from git submodules or might require specific configurations.
-   **Solution**:
    -   The `vendor/` directory is made a Python package by adding `vendor/__init__.py`.
    -   `setup.py` uses `find_packages()` and `include_package_data=True`.
    -   `MANIFEST.in` uses `graft vendor/llama.cpp` to pull in all files from the submodule.

### 2. Excluding Unnecessary Directories

-   **Challenge**: The `vendor/llama.cpp` submodule contains large directories like `models/` and an extensive `examples/` directory that are not needed for the core functionality of the Python package and would bloat its size.
-   **Solution**:
    -   `MANIFEST.in` uses `prune` directives:
        -   `prune vendor/llama.cpp/.git` (removes git metadata)
        -   `prune vendor/llama.cpp/models`
        -   `prune vendor/llama.cpp/examples` (initially removes all original examples)

### 3. Handling CMake Configuration for `llama.cpp`

-   **Challenge**: The `llama.cpp` CMake build system might try to build components (like examples) that we intend to exclude or that might cause issues if their source files are missing from the packaged distribution.
-   **Solution**: The `build_package.py` script automates modifications to the `vendor/llama.cpp/CMakeLists.txt` file *before* the Python packaging commands are run:
    -   It sets the CMake option `LLAMA_BUILD_EXAMPLES` to `OFF`.
    -   It comments out the `add_subdirectory(examples)` line.
    This prevents the `llama.cpp` build system from attempting to compile the examples.

### 4. The `build_package.py` Automation Script

To streamline the build process and ensure consistency, the `build_package.py` script was created. It performs the following key steps in a refined order:

1.  **Initial Setup & Version Reading**:
    *   Reads the current package version from `setup.py`.
2.  **Submodule Processing (Fetch, Clean, Checkout, Clean Again)**:
    *   Ensures the `vendor/llama.cpp` submodule is initialized and updated (`git submodule update --init --recursive`).
    *   Fetches the latest tags from the submodule's remote (`git fetch --tags --force` in `vendor/llama.cpp`).
    *   Identifies the latest relevant release tag (e.g., `bXXXX` pattern).
    *   **Crucially, it now cleans the submodule before attempting a checkout:**
        *   If no new tag is found or the submodule is already on the latest tag, it performs a `git reset --hard <current_commit_or_tag>` and `git clean -fdx` in `vendor/llama.cpp` to ensure a pristine state.
        *   If a new tag needs to be checked out:
            *   It first runs `git reset --hard HEAD` and `git clean -fdx` in `vendor/llama.cpp` to avoid "local changes would be overwritten" errors.
            *   Then, it checks out the `latest_tag`.
            *   Immediately after checkout, it runs `git reset --hard <latest_tag>` and `git clean -fdx` again to ensure the submodule is in the exact, clean state of that tag.
3.  **Modify CMake Configuration**:
    *   *After* the submodule is on its correct, clean commit, the script applies the CMake modifications (disable examples) to `vendor/llama.cpp/CMakeLists.txt`.
4.  **Versioning and Committing to Parent Repository**:
    *   Determines if the submodule pointer in the parent repository changed (e.g., due to a new tag checkout).
    *   If the submodule was updated to a new tag:
        *   It increments the patch version in `setup.py`.
        *   It stages the `setup.py` change and the submodule update (`git add setup.py vendor/llama.cpp`).
        *   It commits these changes to the parent repository with a descriptive message.
    *   If the submodule commit changed for other reasons (e.g., manual update not yet committed), it will also stage and commit the submodule update.
5.  **Clean Build Artifacts**:
    *   Removes previous build artifacts (`dist/`, `build/`, `*.egg-info/`).
6.  **Build Distributions**:
    *   Builds the source distribution (`sdist`) using `python3 setup.py sdist`.
    *   Builds the binary wheel (`bdist_wheel`) using `python3 setup.py bdist_wheel`.

**Using the script:**
```bash
# Ensure the script is executable
chmod +x build_package.py

# Run the script from the project root
python3 build_package.py
```
The resulting distributions will be in the `dist/` directory.

## Important Note on the `examples` Directory in the Package

-   The `build_package.py` script modifies `vendor/llama.cpp/CMakeLists.txt` so that the C++ build system *itself* doesn\'t try to build the examples.
-   `MANIFEST.in` also has `prune vendor/llama.cpp/examples`, which removes the original contents of this directory from the Python package.

-   **Potential Downstream Issue**: Some C++ build systems or parts of the `llama.cpp` CMake setup (even if examples aren\'t built) might still expect an `examples/` directory or a specific file like `examples/CMakeLists.txt` to *exist* in the source tree when the wheel is being installed and the C++ code compiled.

-   **If this becomes an issue**:
    1.  The `vendor/llama.cpp/examples/` directory would need to be recreated as an empty directory containing only a blank `CMakeLists.txt` *before* running `python3 setup.py bdist_wheel` (and `sdist`). This could be added to the `build_package.py` script.
        ```bash
        # Steps to be added to build_package.py before setup.py calls if needed:
        rm -rf vendor/llama.cpp/examples
        mkdir -p vendor/llama.cpp/examples
        touch vendor/llama.cpp/examples/CMakeLists.txt
        ```
    2.  `MANIFEST.in` would then need to be adjusted to specifically include this empty `CMakeLists.txt` file after pruning the original examples:
        ```MANIFEST.in
        graft vendor/llama.cpp
        prune vendor/llama.cpp/.git
        prune vendor/llama.cpp/models
        prune vendor/llama.cpp/examples       # Removes original examples
        include vendor/llama.cpp/examples/CMakeLists.txt # Adds back the empty one
        ```
    As of `build_package.py` creation (2025-05-21), these steps for creating an empty `examples/CMakeLists.txt` are **not** included in the script, as the primary CMake modification should prevent the examples from being built. This note is for future troubleshooting if downstream build issues arise related to a missing `examples` path.

## Current `MANIFEST.in` (as of 2025-05-21)

```plaintext
graft vendor/llama.cpp
prune vendor/llama.cpp/.git
prune vendor/llama.cpp/models
prune vendor/llama.cpp/examples
# recursive-include vendor/llama.cpp * # This line is commented out
```

This setup relies on the `build_package.py` script to prepare the `vendor/llama.cpp/CMakeLists.txt` correctly so that the pruned directories do not cause build failures.
