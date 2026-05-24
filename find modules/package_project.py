"""
Package Project Script
======================
Creates a ZIP archive of the AI Image Classifier project, including all
source files while excluding large/generated artifacts like trained models,
datasets, virtual environments, and caches.

Usage:
    python package_project.py

Output:
    AI_Image_Classifier.zip in the project root directory.
"""

import os
import zipfile
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZIP_FILENAME = "AI_Image_Classifier.zip"

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".py", ".html", ".css", ".js", ".txt", ".md", ".bat", ".sh",
}

# Standalone files to include regardless of extension
INCLUDE_FILES = {
    ".gitignore",
}

# Directories to exclude entirely
EXCLUDE_DIRS = {
    "venv", "env", ".env", ".venv",
    "__pycache__",
    ".git",
    "dataset",
    "uploads",
    ".idea", ".vscode",
    ".ipynb_checkpoints",
    "node_modules",
    "dist", "build",
    "checkpoints", "saved_model",
}

# File extensions to exclude
EXCLUDE_EXTENSIONS = {
    ".h5", ".hdf5", ".pb",
    ".pyc", ".pyo", ".pyd",
    ".egg", ".so",
    ".log",
    ".swp", ".swo",
}

# Specific directories for uploads under static
EXCLUDE_PATHS = {
    os.path.join("static", "uploads"),
}


def should_exclude_dir(dir_name, rel_dir_path):
    """Check if a directory should be excluded from the ZIP."""
    if dir_name in EXCLUDE_DIRS:
        return True
    for exc_path in EXCLUDE_PATHS:
        if rel_dir_path == exc_path or rel_dir_path.startswith(exc_path + os.sep):
            return True
    return False


def should_include_file(filename, rel_file_path):
    """Check if a file should be included in the ZIP."""
    # Exclude the ZIP file itself
    if filename == ZIP_FILENAME:
        return False

    # Exclude by extension
    _, ext = os.path.splitext(filename)
    if ext.lower() in EXCLUDE_EXTENSIONS:
        return False

    # Include standalone files (e.g., .gitignore)
    if filename in INCLUDE_FILES:
        return True

    # Include output PNGs (training graphs, confusion matrix)
    if rel_file_path.startswith("outputs" + os.sep) and ext.lower() == ".png":
        return True

    # Include by extension
    if ext.lower() in INCLUDE_EXTENSIONS:
        return True

    return False


def format_size(size_bytes):
    """Format a byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def package_project():
    """Walk the project tree and create a ZIP archive."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(project_root, ZIP_FILENAME)
    files_added = 0
    files_skipped = 0

    print("=" * 60)
    print("  AI Image Classifier — Project Packager")
    print("=" * 60)
    print(f"\nProject root : {project_root}")
    print(f"Output file  : {ZIP_FILENAME}\n")
    print("-" * 60)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(project_root):
            # Compute relative directory path
            rel_dir = os.path.relpath(dirpath, project_root)
            if rel_dir == ".":
                rel_dir = ""

            # Filter out excluded directories (modifying dirnames in-place
            # prevents os.walk from descending into them)
            dirnames[:] = [
                d for d in dirnames
                if not should_exclude_dir(d, os.path.join(rel_dir, d) if rel_dir else d)
            ]

            for filename in filenames:
                rel_file_path = os.path.join(rel_dir, filename) if rel_dir else filename

                if should_include_file(filename, rel_file_path):
                    abs_file_path = os.path.join(dirpath, filename)
                    # Store with a top-level folder name inside the ZIP
                    arcname = os.path.join("AI_Image_Classifier", rel_file_path)
                    zf.write(abs_file_path, arcname)
                    file_size = os.path.getsize(abs_file_path)
                    print(f"  [OK] {rel_file_path:<50s} {format_size(file_size):>10s}")
                    files_added += 1
                else:
                    files_skipped += 1

    zip_size = os.path.getsize(zip_path)

    print("-" * 60)
    print(f"\n  Files included : {files_added}")
    print(f"  Files skipped  : {files_skipped}")
    print(f"  ZIP size       : {format_size(zip_size)}")
    print(f"  Saved to       : {zip_path}")
    print("\n" + "=" * 60)
    print("  [SUCCESS] Packaging complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        package_project()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}", file=sys.stderr)
        sys.exit(1)
