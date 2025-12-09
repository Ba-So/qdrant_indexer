"""Sample Python module for testing code loaders.

This module contains various Python constructs including classes, methods,
functions, and constants to comprehensively test code parsing.
"""

# Module-level constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30


def helper_function(value: int) -> int:
    """A simple helper function that doubles a value.

    Args:
        value: The integer value to double.

    Returns:
        The doubled value.
    """
    return value * 2


class DataProcessor:
    """Process data with various operations.

    This class demonstrates a typical Python class structure with
    methods, properties, and documentation.
    """

    def __init__(self, name: str):
        """Initialize the processor.

        Args:
            name: The name of this processor instance.
        """
        self.name = name
        self._cache = {}

    def process(self, data: str) -> str:
        """Process input data.

        Args:
            data: The data string to process.

        Returns:
            The processed data string.
        """
        return data.upper()

    def cached_process(self, data: str) -> str:
        """Process data with caching.

        Args:
            data: The data to process.

        Returns:
            Processed data from cache or newly processed.
        """
        if data not in self._cache:
            self._cache[data] = self.process(data)
        return self._cache[data]

    @property
    def cache_size(self) -> int:
        """Get the current cache size.

        Returns:
            Number of items in the cache.
        """
        return len(self._cache)


class AdvancedProcessor(DataProcessor):
    """An advanced processor with additional capabilities.

    Extends DataProcessor with extra functionality.
    """

    def process(self, data: str) -> str:
        """Process data with advanced techniques.

        Overrides parent process method.

        Args:
            data: The data to process.

        Returns:
            Advanced processed data.
        """
        result = super().process(data)
        return f"[ADVANCED] {result}"
