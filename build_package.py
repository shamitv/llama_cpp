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
                new_lines.append('option(LLAMA_BUILD_EXAMPLES "llama: build examples" OFF) # Modified by build script\n')
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
        return

    effective_version_for_build = current_version
    submodule_path_for_git_add = os.path.relpath(LLAMA_CPP_SUBMODULE_PATH, PROJECT_ROOT)

    # --- Step 1: Submodule Operations (Fetch, Checkout, Clean) ---
    print("\\n--- Step 1: Processing submodule (Fetch, Checkout, Clean) ---")
    try:
        run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT)
        run_command(["git", "fetch", "--tags", "--force"], cwd=LLAMA_CPP_SUBMODULE_PATH)

        tag_discovery_result = run_command(
            "git tag -l | grep -E '^b[0-9]+$' | sort -V | tail -n 1 || true",
            cwd=LLAMA_CPP_SUBMODULE_PATH, shell=True
        )
        if tag_discovery_result.stderr:
            print(f"Stderr from tag discovery command: {tag_discovery_result.stderr.strip()}")
        latest_tag = tag_discovery_result.stdout.strip()
        print(f"Raw output from tag discovery (stdout): '{latest_tag}'")

        current_submodule_commit_before_ops = run_command(["git", "rev-parse", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH).stdout.strip()
        tag_checked_out_in_this_run = None

        if not latest_tag:
            print("No matching bXXXX release tags found. Submodule will use its current commit.")
            # Ensure current state is clean before CMake modification
            print("Resetting submodule to current HEAD and cleaning...")
            run_command(["git", "reset", "--hard", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH)
            run_command(["git", "clean", "-fdx"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        else:
            print(f"Latest suitable tag found: {latest_tag}")
            current_exact_tag_result = run_command(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH, check=False)
            current_exact_tag = current_exact_tag_result.stdout.strip() if current_exact_tag_result.returncode == 0 else None

            if current_exact_tag == latest_tag:
                print(f"Submodule already at tag {latest_tag}. Resetting to tag's state and cleaning.")
                run_command(["git", "reset", "--hard", latest_tag], cwd=LLAMA_CPP_SUBMODULE_PATH)
                run_command(["git", "clean", "-fdx"], cwd=LLAMA_CPP_SUBMODULE_PATH)
            else:
                print(f"Submodule not on tag {latest_tag} (current: {current_exact_tag or current_submodule_commit_before_ops[:7]}).")
                print("Resetting submodule to current HEAD, cleaning, then checking out tag...")
                run_command(["git", "reset", "--hard", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH) # Clean before checkout
                run_command(["git", "clean", "-fdx"], cwd=LLAMA_CPP_SUBMODULE_PATH)
                
                run_command(["git", "checkout", latest_tag], cwd=LLAMA_CPP_SUBMODULE_PATH) # Checkout the tag
                
                # After checkout, ensure it's on the tag's pristine state
                run_command(["git", "reset", "--hard", latest_tag], cwd=LLAMA_CPP_SUBMODULE_PATH) 
                run_command(["git", "clean", "-fdx"], cwd=LLAMA_CPP_SUBMODULE_PATH)
                tag_checked_out_in_this_run = latest_tag
                print(f"Successfully checked out and cleaned tag {latest_tag}.")
        
        # --- Step 2: Modify CMake in submodule ---
        # This happens AFTER submodule is on its intended commit (tag or previous) and is clean.
        print("\\n--- Step 2: Modifying CMake configuration in submodule ---")
        modify_cmake_config() # This function prints whether it made changes or not

        # --- Step 3: Versioning and Committing to Parent Repo ---
        print("\\n--- Step 3: Handling versioning and committing to parent repository ---")
        final_submodule_commit_after_ops = run_command(["git", "rev-parse", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH).stdout.strip()
        
        submodule_commit_changed_in_parent = (current_submodule_commit_before_ops != final_submodule_commit_after_ops)
        version_was_incremented = False
        commit_actions_taken = []


        if tag_checked_out_in_this_run: # Implies submodule_commit_changed_in_parent is true
            action_msg = f"Update {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)} to {tag_checked_out_in_this_run}"
            print(action_msg)
            commit_actions_taken.append(action_msg)

            version_parts = current_version.split('.')
            if len(version_parts) >= 3:
                try:
                    version_parts[-1] = str(int(version_parts[-1]) + 1)
                    potential_new_version = ".".join(version_parts)
                    update_version_in_setup_py(SETUP_PY_PATH, potential_new_version, current_version)
                    effective_version_for_build = potential_new_version
                    version_was_incremented = True
                    commit_actions_taken.append(f"bump version to {effective_version_for_build}")
                except ValueError:
                    print(f"Warning: Could not parse/increment patch version for '{current_version}'.")
            else:
                print(f"Warning: Version '{current_version}' from setup.py does not have at least 3 parts (X.Y.Z). Not incremented.")
        
        elif submodule_commit_changed_in_parent: # Commit changed but not due to a new tag checkout by this script
            action_msg = f"Sync {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)} to {final_submodule_commit_after_ops[:7]}"
            print(action_msg)
            commit_actions_taken.append(action_msg)
        
        # If CMakeLists.txt was modified by modify_cmake_config(), the submodule is "dirty".
        # The parent repo needs to record the new state of the submodule (its new commit if CMake changes were committed inside, or just its current commit).
        # Our script does not commit *inside* the submodule. So, if modify_cmake_config changed CMakeLists.txt,
        # the submodule's working tree is dirty. `git add vendor/llama.cpp` in parent will stage the submodule's current commit.
        # If that commit doesn't include the CMakeLists.txt changes, those changes are "lost" from parent's perspective unless committed inside submodule.
        # For now, we assume the state after modify_cmake_config is what we want associated with the submodule's current HEAD.
        # The `git add submodule_path_for_git_add` will stage the current HEAD of the submodule.

        if submodule_commit_changed_in_parent or version_was_incremented:
            print("Staging changes in parent repository...")
            run_command(["git", "add", submodule_path_for_git_add], cwd=PROJECT_ROOT)
            if version_was_incremented:
                run_command(["git", "add", SETUP_PY_PATH], cwd=PROJECT_ROOT)
            
            final_commit_message = ", ".join(filter(None, commit_actions_taken))
            if not final_commit_message: # Fallback
                final_commit_message = f"Automated update for {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)}"

            status_result_before_commit = run_command(["git", "status", "--porcelain"], cwd=PROJECT_ROOT)
            if status_result_before_commit.stdout.strip(): # Check if there are staged changes
                run_command(["git", "commit", "-m", final_commit_message], cwd=PROJECT_ROOT)
                print(f"Committed to parent repo: {final_commit_message}")
            else:
                print("No changes were staged in the parent repository for commit, though actions were recorded.")
        else:
            print(f"No changes to submodule pointer or setup.py requiring a commit to parent repo.")
            print(f"Submodule {os.path.basename(LLAMA_CPP_SUBMODULE_PATH)} remains at {final_submodule_commit_after_ops[:7]}.")

        print("Submodule update, CMake modification, and versioning process finished.")

    except subprocess.CalledProcessError as e:
        print(f"ERROR during script execution: {e}")
        if hasattr(e, 'stdout') and e.stdout: print(f"Stdout: {e.stdout.strip()}")
        if hasattr(e, 'stderr') and e.stderr: print(f"Stderr: {e.stderr.strip()}")
        print("Build will continue, but its state may be inconsistent or based on previous configurations.")
    except Exception as e:
        print(f"UNEXPECTED ERROR during script execution: {e}")
        import traceback
        traceback.print_exc()
        print("Build will continue, but its state may be inconsistent or based on previous configurations.")
    
    print(f"\\n--- Step 4: Building the wheel (using version: {effective_version_for_build}) ---")
    # --- 4. Generate source and binary wheel ---
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
