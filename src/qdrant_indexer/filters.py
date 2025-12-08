"""File filtering utilities for excluding files during indexing."""

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default patterns to exclude
DEFAULT_EXCLUDE_PATTERNS = [
    "node_modules/**",
    ".git/**",
    "__pycache__/**",
    "*.pyc",
    "venv/**",
    ".venv/**",
    ".tox/**",
    ".nox/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "*.egg-info/**",
    "dist/**",
    "build/**",
    ".direnv/**",
]


def should_exclude(file_path: Path, base_path: Path, exclude_patterns: list[str]) -> bool:
    """Check if a file should be excluded based on patterns.

    Args:
        file_path: Absolute path to the file.
        base_path: Base directory being indexed.
        exclude_patterns: List of glob patterns to exclude.

    Returns:
        True if the file should be excluded, False otherwise.
    """
    # Get relative path for pattern matching
    try:
        rel_path = file_path.relative_to(base_path)
    except ValueError:
        rel_path = file_path

    rel_str = str(rel_path)

    for pattern in exclude_patterns:
        # Check if any part of the path matches
        if fnmatch.fnmatch(rel_str, pattern):
            logger.debug(f"Excluding {rel_str}: matches pattern '{pattern}'")
            return True

        # Also check individual path components
        for part in rel_path.parts:
            if fnmatch.fnmatch(part, pattern):
                logger.debug(f"Excluding {rel_str}: component '{part}' matches pattern '{pattern}'")
                return True

        # Check if any parent directory matches (for patterns like "node_modules/**")
        if pattern.endswith("/**"):
            dir_pattern = pattern[:-3]
            for part in rel_path.parts[:-1]:  # Exclude the filename itself
                if fnmatch.fnmatch(part, dir_pattern):
                    logger.debug(f"Excluding {rel_str}: parent '{part}' matches pattern '{dir_pattern}'")
                    return True

    return False


def filter_files(
    files: list[Path],
    base_path: Path,
    exclude_patterns: list[str] | None = None,
    use_defaults: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Filter a list of files based on exclusion patterns.

    Args:
        files: List of file paths to filter.
        base_path: Base directory being indexed.
        exclude_patterns: Additional patterns to exclude.
        use_defaults: Whether to include default exclusion patterns.

    Returns:
        Tuple of (included_files, excluded_files).
    """
    patterns = []

    if use_defaults:
        patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

    if exclude_patterns:
        patterns.extend(exclude_patterns)

    included = []
    excluded = []

    for file_path in files:
        if should_exclude(file_path, base_path, patterns):
            excluded.append(file_path)
        else:
            included.append(file_path)

    if excluded:
        logger.info(f"Excluded {len(excluded)} files based on patterns")

    return included, excluded
