\
import subprocess
import os
import shutil
import re # For regex operations on setup.py

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in project root
LLAMA_CPP_SUBMODULE_PATH = os.path.join(PROJECT_ROOT, "vendor", "llama.cpp")
LLAMA_CPP_EXAMPLES_PATH = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "examples")
LLAMA_CPP_MODELS_PATH = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "models")
LLAMA_CPP_CMAKE_FILE = os.path.join(LLAMA_CPP_SUBMODULE_PATH, "CMakeLists.txt")
SETUP_PY_PATH = os.path.join(PROJECT_ROOT, "setup.py")

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

def get_current_version(file_path):
    """Reads setup.py and extracts the version."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
        raise ValueError(f"Could not find version pattern in {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {file_path} not found.")
    except Exception as e:
        raise Exception(f"Error reading version from {file_path}: {e}")

def update_version_in_setup_py(file_path, new_version, old_version):
    """Writes the new version to setup.py."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        old_version_pattern = r"version\s*=\s*['\"]" + re.escape(old_version) + r"['\"]"
        new_version_string = f"version='{new_version}'" # Standardize on single quotes for the new version
        
        if not re.search(old_version_pattern, content):
            print(f"Warning: Old version '{old_version}' not found in {file_path} as expected using pattern '{old_version_pattern}'. Version not updated.")
            return

        new_content, num_replacements = re.subn(old_version_pattern, new_version_string, content, count=1)
        
        if num_replacements == 0:
            print(f"Warning: Old version '{old_version}' was not replaced in {file_path}. Check the version string and pattern.")
            return

        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Updated version in {file_path} from {old_version} to {new_version}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found during version update.")
    except Exception as e:
        print(f"Error updating version in {file_path}: {e}")


def main():
    os.chdir(PROJECT_ROOT) # Ensure commands run from project root

    try:
        current_version = get_current_version(SETUP_PY_PATH)
        print(f"Current package version from setup.py: {current_version}")
    except Exception as e:
        print(f"Fatal error: Could not retrieve current version from setup.py. {e}")
        return # Exit if we can't get the version

    # This variable will hold the version that should be used by the build.
    # It starts as current_version and might be updated if a new tag is processed.
    effective_version_for_build = current_version

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
                print(f"Submodule updated from {current_submodule_commit[:7]} to {new_submodule_commit[:7]} (tag {latest_tag}).")
                
                version_changed_this_run = False
                potential_new_version = current_version # Start with current version

                # Try to increment version
                version_parts = current_version.split('.')
                if len(version_parts) >= 3: # Expecting at least X.Y.Z
                    try:
                        version_parts[-1] = str(int(version_parts[-1]) + 1)
                        potential_new_version = ".".join(version_parts)
                        version_changed_this_run = True
                    except ValueError:
                        print(f"Warning: Could not parse and increment patch version for '{current_version}'. Version not incremented.")
                else:
                    print(f"Warning: Version '{current_version}' from setup.py does not have at least 3 parts (X.Y.Z). Version not incremented.")
                
                # Stage submodule update
                run_command(["git", "add", os.path.relpath(LLAMA_CPP_SUBMODULE_PATH, PROJECT_ROOT)], cwd=PROJECT_ROOT)
                commit_message_parts = [f"Update {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)} to {latest_tag}"]

                if version_changed_this_run:
                    update_version_in_setup_py(SETUP_PY_PATH, potential_new_version, current_version) # Write to file
                    run_command(["git", "add", SETUP_PY_PATH], cwd=PROJECT_ROOT) # Add setup.py
                    commit_message_parts.append(f"bump version to {potential_new_version}")
                    effective_version_for_build = potential_new_version # Update the version for this build run
                
                final_commit_message = ", ".join(commit_message_parts)
                
                # Check if there are staged changes before committing
                status_result = run_command(["git", "status", "--porcelain"], cwd=PROJECT_ROOT)
                if status_result.stdout.strip(): # Ensure there's something to commit
                    run_command(["git", "commit", "-m", final_commit_message], cwd=PROJECT_ROOT)
                    print(f"Committed: {final_commit_message}")
                else:
                    # This case might happen if the submodule was already on the tag,
                    # but the main repo hadn't committed that submodule state yet.
                    print("No changes staged for commit, or submodule pointer was already up-to-date and committed.")
            else:
                print(f"Submodule {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)} already at tag {latest_tag}.")
        print("Submodule update process finished.")
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

    print(f"Proceeding to build with version: {effective_version_for_build}")
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
