# Llama CPP

This is a Python package for Llama CPP ( https://github.com/ggml-org/llama.cpp ).

## Updating the `llama.cpp` Submodule

To update the `vendor/llama.cpp` submodule to a new release tag, follow these steps:

1.  Navigate to the submodule directory:
    ```bash
    cd vendor/llama.cpp
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
    git add vendor/llama.cpp
    ```

6.  Commit the update in the main project:
    ```bash
    git commit -m "Update llama.cpp submodule to tag new_tag_name"
    ```

This will ensure that the main project now points to the updated version of the `llama.cpp` submodule.
