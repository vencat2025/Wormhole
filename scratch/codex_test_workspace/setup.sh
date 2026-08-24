#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display usage information.
usage() {
    echo "Usage: $0 [-r requirements.txt] [-p package1 package2 ...]"
    echo "  -r  Path to a requirements file (default: requirements.txt)"
    echo "  -p  List of additional packages to install via pip"
    exit 1
}

# Default values
REQ_FILE="requirements.txt"
EXTRA_PACKAGES=()

# Parse command‑line options.
while getopts ":r:p:" opt; do
    case $opt in
        r) REQ_FILE="$OPTARG" ;;
        p) EXTRA_PACKAGES+=("$OPTARG") ;;
        *) usage ;;
    esac
done
shift $((OPTIND -1))

# If there are remaining arguments after options, treat them as extra packages.
if [[ $# -gt 0 ]]; then
    EXTRA_PACKAGES+=("$@")
fi

# Ensure pip is available; install it if missing.
if ! command -v pip &>/dev/null; then
    echo "pip not found. Installing pip..."
    python -m ensurepip --upgrade
fi

# Upgrade pip to the latest version.
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install packages from the requirements file if it exists.
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing packages from $REQ_FILE..."
    pip install -r "$REQ_FILE"
else
    echo "No requirements file found at $REQ_FILE."
fi

# Install any extra packages specified.
if [[ ${#EXTRA_PACKAGES[@]} -gt 0 ]]; then
    echo "Installing extra packages: ${EXTRA_PACKAGES[*]}..."
    pip install "${EXTRA_PACKAGES[@]}"
fi

echo "Setup complete."
