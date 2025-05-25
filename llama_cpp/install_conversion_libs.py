import sys
import subprocess

# List of required python libraries and their versions for HF to GGUF conversion
REQUIRED_CONVERSION_LIBRARIES = [
    "transformers>=4.0.0",
    "numpy>=1.20.0",
    "torch>=1.10.0",
    "safetensors>=0.3.0",
]

def install_conversion_libs():
    """
    Installs all python libraries required to invoke the HF to GGUF conversion script.

    Returns:
        bool: True if all installations succeed, False otherwise.
    """
    success = True
    for package in REQUIRED_CONVERSION_LIBRARIES:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            success = False
    return success
