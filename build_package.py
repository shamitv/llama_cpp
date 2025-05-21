\
import subprocess
import os
import shutil

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in project root
LLAMA_CPP_SUBMODULE_PATH = os.path.join(PROJECT_ROOT, "vendor", "llama.cpp")
LLAMA_CPP_EXAMPLES_PATH = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "examples")
LLAMA_CPP_MODELS_PATH = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "models")
LLAMA_CPP_CMAKE_FILE = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "CMakeLists.txt")

def run_command(command, cwd=None, check=True, shell=False):
    """Helper function to run a shell command."""
    print(f"Running command: {command if shell else ' '.join(command)} (in {cwd or PROJECT_ROOT})")
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, shell=shell)
    if check and process.returncode != 0:
        print(f"Error running command: {command if shell else ' '.join(command)}")
        print(f"Stdout: {process.stdout}")
        print(f"Stderr: {process.stderr}")
        raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout, stderr=process.stderr)
    return process

def modify_cmake_config():
    """
    Modifies the CMakeLists.txt in vendor/llama.cpp to:
    1. Set LLAMA_BUILD_EXAMPLES to OFF.
    2. Comment out `add_subdirectory(examples)`.
    """
    print(f"Modifying CMake config: {LLAMA_CPP_CMAKE_FILE}")
    if not os.path.exists(LLAMA_CPP_CMAKE_FILE):
        print(f"Error: CMake file not found at {LLAMA_CPP_CMAKE_FILE}")
        return

    with open(LLAMA_CPP_CMAKE_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    modified = False
    for line in lines:
        if "option(LLAMA_BUILD_EXAMPLES" in line:
            if "OFF" not in line:
                new_lines.append('option(LLAMA_BUILD_EXAMPLES "llama: build examples" OFF) # Modified by build script\\n')
                print(" - Set LLAMA_BUILD_EXAMPLES to OFF")
                modified = True
            else:
                new_lines.append(line) # Already off
        elif "add_subdirectory(examples)" in line and not line.strip().startswith("#"):
            new_lines.append("# " + line) # Comment out adding the examples subdirectory
            print(" - Commented out add_subdirectory(examples)")
            modified = True
        else:
            new_lines.append(line)

    if modified:
        with open(LLAMA_CPP_CMAKE_FILE, 'w') as f:
            f.writelines(new_lines)
        print("CMake config modified.")
    else:
        print("CMake config already up-to-date.")

def main():
    os.chdir(PROJECT_ROOT) # Ensure commands run from project root

    # --- 1. Update submodule to latest release tag ---
    print("\\n--- Step 1: Updating submodule to latest release tag ---")
    try:
        run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT) # Ensure submodule is initialized
        run_command(["git", "fetch", "--tags", "--force"], cwd=LLAMA_CPP_SUBMODULE_PATH) # Force fetch tags

        # Get the latest tag name (sorts tags by version and picks the last one)
        # This assumes semantic versioning (vX.Y.Z or X.Y.Z)
        result = run_command(
            "git tag -l | sort -V | tail -n 1",
            cwd=LLAMA_CPP_SUBMODULE_PATH,
            shell=True # sort and tail are shell operations
        )
        latest_tag = result.stdout.strip()

        if not latest_tag:
            print("No tags found in submodule. Skipping tag checkout.")
        else:
            print(f"Latest tag found: {latest_tag}")
            current_submodule_commit = run_command(["git", "rev-parse", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH).stdout.strip()
            run_command(["git", "checkout", latest_tag], cwd=LLAMA_CPP_SUBMODULE_PATH)
            new_submodule_commit = run_command(["git", "rev-parse", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH).stdout.strip()

            if current_submodule_commit != new_submodule_commit:
                run_command(["git", "add", os.path.relpath(LLAMA_CPP_SUBMODULE_PATH, PROJECT_ROOT)], cwd=PROJECT_ROOT)
                commit_message = f"Update {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)} submodule to tag {latest_tag}"
                # Check if there are staged changes before committing
                status_result = run_command(["git", "status", "--porcelain"], cwd=PROJECT_ROOT)
                if status_result.stdout:
                    run_command(["git", "commit", "-m", commit_message], cwd=PROJECT_ROOT)
                    print(f"Committed update for submodule to tag {latest_tag}.")
                else:
                    print("No changes to commit regarding submodule pointer.")
            else:
                print(f"Submodule already at tag {latest_tag}.")
        print("Submodule updated.")
    except subprocess.CalledProcessError as e:
        print(f"Error during submodule update: {e}")
        print("Continuing with the build, but submodule might not be at the latest tag.")
    except Exception as e:
        print(f"An unexpected error occurred during submodule update: {e}")
        print("Continuing with the build, but submodule might not be at the latest tag.")


    # --- 2. Fetch current source from git submodule (Reset any local changes) ---
    print("\\n--- Step 2: Resetting submodule to fetched state ---")
    try:
        run_command(["git", "reset", "--hard", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        run_command(["git", "clean", "-fdx"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        print("Submodule reset to clean state.")
    except subprocess.CalledProcessError as e:
        print(f"Error during submodule reset: {e}")
        # Decide if this is a fatal error or if you can continue
        return # Or raise

    # --- 3. Modify CMake config so that it does not expect examples directory ---
    print("\\n--- Step 3: Modifying CMake configuration ---")
    try:
        modify_cmake_config()
    except Exception as e:
        print(f"Error modifying CMake config: {e}")
        return # Or raise

    # --- 4. Generate source and binary wheel ---
    print("\\n--- Step 4: Building the wheel ---")
    # Clean previous builds
    print("Cleaning previous build artifacts...")
    if os.path.exists(os.path.join(PROJECT_ROOT, "dist")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "dist"))
    if os.path.exists(os.path.join(PROJECT_ROOT, "build")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "build"))
    
    egg_info_dirs = [d for d in os.listdir(PROJECT_ROOT) if d.endswith(".egg-info")]
    for egg_dir in egg_info_dirs:
        shutil.rmtree(os.path.join(PROJECT_ROOT, egg_dir))

    # Build the wheel
    try:
        print("Building sdist...")
        run_command(["python3", "setup.py", "sdist"], cwd=PROJECT_ROOT)
        print("Building bdist_wheel...")
        run_command(["python3", "setup.py", "bdist_wheel"], cwd=PROJECT_ROOT)
        print("Wheel build process completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during wheel build: {e}")
        return # Or raise
    
    print("\\n--- Automation Script Finished ---")
    print(f"Check the '{os.path.join(PROJECT_ROOT, 'dist')}' directory for the generated wheel and source distribution.")

if __name__ == "__main__":
    main()
