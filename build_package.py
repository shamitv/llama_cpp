\
import os
import subprocess
import re # For regex operations on setup.py
import urllib.request
import shutil
import glob # For finding .egg-info directories

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in project root
LLAMA_CPP_SUBMODULE_PATH = os.path.join(PROJECT_ROOT, "vendor", "llama.cpp")
LLAMA_CPP_PACKAGE_BINARIES_PATH = os.path.join(PROJECT_ROOT, "llama_cpp", "binaries") # For storing downloaded binaries
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

def download_and_place_windows_binary(tag_name):
    """Downloads the Windows binary for the given tag and places it in the package."""
    print(f"\\n--- Downloading Windows binary for tag: {tag_name} ---")
    os.makedirs(LLAMA_CPP_PACKAGE_BINARIES_PATH, exist_ok=True)

    # Clear previous binaries in llama_cpp/binaries
    print(f"Cleaning previous binaries from {LLAMA_CPP_PACKAGE_BINARIES_PATH}...")
    for item in os.listdir(LLAMA_CPP_PACKAGE_BINARIES_PATH):
        item_path = os.path.join(LLAMA_CPP_PACKAGE_BINARIES_PATH, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


    # Construct binary filename and URL
    # Example tag from user: b5479. Releases use full tags like 'master-b5479' or 'b5479'
    # We need to handle if the tag from `git describe` has a prefix or not.
    # The release asset seems to use the plain tag number.
    # Let's assume the tag_name from `git describe --tags --abbrev=0` is what's needed for the URL path,
    # but the filename might need processing if it contains prefixes not in the release asset name.
    
    # Try to extract the 'bXXXX' part if it's a longer tag like 'master-bXXXX'
    match = re.search(r'(b\d+[a-fA-F0-9]*)$', tag_name)
    if match:
        processed_tag_name_for_file = match.group(1)
    else:
        processed_tag_name_for_file = tag_name # Use as is if no clear 'bXXXX' pattern

    binary_filename = f"llama-{processed_tag_name_for_file}-bin-win-cpu-x64.zip"
    # The download URL uses the full tag name as it appears in the releases page
    download_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{binary_filename}"
    binary_dest_path = os.path.join(LLAMA_CPP_PACKAGE_BINARIES_PATH, binary_filename)

    print(f"Attempting to download: {download_url}")
    try:
        # Create a request object to disable SSL verification if needed, or set headers
        req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(binary_dest_path, 'wb') as out_file:
            if response.status == 200:
                shutil.copyfileobj(response, out_file)
                print(f"Successfully downloaded {binary_filename} to {binary_dest_path}")
            else:
                print(f"Warning: Failed to download {binary_filename}. Status: {response.status}")
                print("Windows binary will not be included.")
                if os.path.exists(binary_dest_path): os.remove(binary_dest_path)
    except urllib.error.HTTPError as e:
        print(f"Warning: HTTPError {e.code} when trying to download {binary_filename} from {download_url}: {e.reason}")
        print("This could mean the binary for this specific tag/format doesn't exist or the URL is incorrect.")
        print("Windows binary will not be included.")
        if os.path.exists(binary_dest_path): os.remove(binary_dest_path)
    except urllib.error.URLError as e:
        print(f"Warning: URLError when trying to download {binary_filename}: {e.reason}")
        print("This could be a network issue or an invalid URL.")
        print("Windows binary will not be included.")
        if os.path.exists(binary_dest_path): os.remove(binary_dest_path)
    except Exception as e:
        print(f"Warning: An unexpected error occurred while downloading {binary_filename}: {e}")
        print("Windows binary will not be included.")
        if os.path.exists(binary_dest_path): os.remove(binary_dest_path)

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

    current_version = "0.0.0" # Fallback
    try:
        current_version = get_current_version(SETUP_PY_PATH)
        print(f"--- Current package version from setup.py: {current_version} ---")
    except Exception as e:
        print(f"Warning: Could not read version from setup.py: {e}. Using fallback '{current_version}'.")

    effective_version_for_build = current_version
    # submodule_path_for_git_add = os.path.relpath(LLAMA_CPP_SUBMODULE_PATH, PROJECT_ROOT) # Not used currently

    # --- Step 1: Submodule Operations (Fetch, Checkout, Clean) ---
    print("\\n--- Step 1: Processing submodule (Fetch, Checkout, Clean) ---")
    submodule_tag = None # Initialize submodule_tag
    try:
        run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT)
        
        # Update submodule to the latest from remote
        print(f"Fetching latest changes for submodule at {LLAMA_CPP_SUBMODULE_PATH}...")
        run_command(["git", "fetch"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        print(f"Checking out master branch for submodule at {LLAMA_CPP_SUBMODULE_PATH}...")
        run_command(["git", "checkout", "master"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        print(f"Resetting submodule at {LLAMA_CPP_SUBMODULE_PATH} to origin/master (discarding local changes)...")
        run_command(["git", "reset", "--hard", "origin/master"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        print(f"Pulling latest changes for submodule at {LLAMA_CPP_SUBMODULE_PATH}...")
        run_command(["git", "pull"], cwd=LLAMA_CPP_SUBMODULE_PATH)
        print(f"Fetching all tags for submodule at {LLAMA_CPP_SUBMODULE_PATH}...")
        run_command(["git", "fetch", "--tags"], cwd=LLAMA_CPP_SUBMODULE_PATH)

        # Get current submodule commit and the latest tag on that commit
        # Using --abbrev=0 to get the plain tag name like 'bxxxx' or 'vX.Y.Z'
        # Using --dirty to check if the submodule has local modifications
        # Using --always to ensure a commit hash is returned if no tag is found
        git_describe_cmd = ["git", "describe", "--tags", "--abbrev=0", "--always"]
        process_result = run_command(git_describe_cmd, cwd=LLAMA_CPP_SUBMODULE_PATH)
        submodule_tag = process_result.stdout.strip()

        if not submodule_tag:
            print("Warning: Could not determine a tag for the submodule. Will try with commit hash.")
            # Fallback to commit hash if no tag
            process_result = run_command(["git", "rev-parse", "--short", "HEAD"], cwd=LLAMA_CPP_SUBMODULE_PATH)
            submodule_tag = process_result.stdout.strip()

        print(f"Submodule vendor/llama.cpp is at ref: {submodule_tag}")

        # Check if submodule is dirty
        status_process = run_command(["git", "status", "--porcelain"], cwd=LLAMA_CPP_SUBMODULE_PATH, check=False)
        if status_process.stdout.strip():
            print(f"Warning: Submodule {LLAMA_CPP_SUBMODULE_PATH} is dirty. This might affect the build.")
            # print(status_process.stdout) # Optionally print dirty status

    except subprocess.CalledProcessError as e:
        print(f"Error during submodule operations: {e}")
        print("Cannot proceed without submodule information. Exiting.")
        return # Exit if submodule operations fail critically
    except Exception as e:
        print(f"An unexpected error occurred during submodule operations: {e}")
        print("Cannot proceed without submodule information. Exiting.")
        return # Exit if submodule operations fail critically

    # --- Step 1.5: Download Windows Binary ---
    if submodule_tag:
        download_and_place_windows_binary(submodule_tag)
    else:
        print("Critical: Submodule tag/commit not determined. Skipping download of Windows binary.")
        # Ensure binaries directory exists for setup.py even if download fails, but it should be there from download_and_place_windows_binary
        os.makedirs(LLAMA_CPP_PACKAGE_BINARIES_PATH, exist_ok=True)


    # --- Step 2: Modify CMake Configuration in submodule (if needed) ---
    print("\\n--- Step 2: Modifying CMake config in submodule (if applicable) ---")
    # modify_cmake_config() # Call your CMake modification function if it's still needed

    # --- Step 3: Versioning and Committing to Parent Repository (Simplified) ---
    # This section is simplified. Original script had more complex logic for version bumping
    # and committing submodule changes. For now, we assume the parent repo is managed manually
    # regarding submodule pointer updates.
    print("\\n--- Step 3: Versioning and Committing (Simplified) ---")
    print(f"Parent repository operations (like committing submodule changes) are assumed to be handled manually or by a separate process for now.")


    print(f"\\n--- Step 4: Building the wheel (using version: {effective_version_for_build}) ---")
    # --- 4. Generate source and binary wheel ---
    # Clean previous builds
    print("Cleaning previous build artifacts...")
    if os.path.exists(os.path.join(PROJECT_ROOT, "dist")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "dist"))
    if os.path.exists(os.path.join(PROJECT_ROOT, "build")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "build"))
    
    # Use glob to find and remove .egg-info directories
    egg_info_dirs = glob.glob(os.path.join(PROJECT_ROOT, "*.egg-info"))
    for egg_dir in egg_info_dirs:
        print(f"Removing old .egg-info directory: {egg_dir}")
        shutil.rmtree(egg_dir)

    # Build the source distribution and wheel
    try:
        print("Building source distribution (sdist)...")
        run_command(["python3", SETUP_PY_PATH, "sdist"], cwd=PROJECT_ROOT)
        print("Building wheel (bdist_wheel)...")
        run_command(["python3", SETUP_PY_PATH, "bdist_wheel"], cwd=PROJECT_ROOT)
        print("Wheel build process completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during wheel build: {e}")
        return # Or raise
    
    print("\\n--- Automation Script Finished ---")
    print(f"Check the '{os.path.join(PROJECT_ROOT, 'dist')}' directory for the generated wheel and source distribution.")

if __name__ == "__main__":
    main()
