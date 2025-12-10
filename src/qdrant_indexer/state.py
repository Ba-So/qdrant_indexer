"""State management for incremental indexing."""

import hashlib
import json
from pathlib import Path

from qdrant_indexer.models import IndexedFileState


class IndexState:
    """Manages persistent index state for incremental updates."""

    def __init__(self, state_file: Path):
        """Initialize IndexState with a path to the state file.

        Args:
            state_file: Path to the JSON file storing index state.
        """
        self.state_file = state_file
        self.files: dict[str, IndexedFileState] = {}

    def load(self) -> None:
        """Load state from JSON file."""
        if not self.state_file.exists():
            return
        with open(self.state_file, "r") as f:
            data = json.load(f)
            self.files = {
                path: IndexedFileState(**state_data)
                for path, state_data in data.get("files", {}).items()
            }

    def save(self) -> None:
        """Persist state to JSON file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "files": {
                path: {
                    "path": state.path,
                    "content_hash": state.content_hash,
                    "indexed_at": state.indexed_at,
                    "chunk_count": state.chunk_count,
                    "chunk_ids": state.chunk_ids,
                }
                for path, state in self.files.items()
            }
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_file_state(self, path: Path) -> IndexedFileState | None:
        """Get state for a file.

        Args:
            path: Path to the file.

        Returns:
            IndexedFileState if found, None otherwise.
        """
        return self.files.get(str(path.absolute()))

    def set_file_state(self, path: Path, state: IndexedFileState) -> None:
        """Update state for a file.

        Args:
            path: Path to the file.
            state: New state for the file.
        """
        self.files[str(path.absolute())] = state

    def remove_file(self, path: Path) -> None:
        """Remove a file from state.

        Args:
            path: Path to the file to remove.
        """
        abs_path = str(path.absolute())
        if abs_path in self.files:
            del self.files[abs_path]

    def get_all_paths(self) -> set[str]:
        """Get all tracked file paths.

        Returns:
            Set of absolute file paths.
        """
        return set(self.files.keys())


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash of the file content.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
