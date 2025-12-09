<?php
/**
 * Sample PHP file for testing code loaders.
 *
 * This file contains various PHP constructs including classes, methods,
 * functions, and different visibility modifiers.
 */

/**
 * A helper function that doubles a value.
 *
 * @param int $value The value to double
 * @return int The doubled value
 */
function helper_function($value) {
    return $value * 2;
}

/**
 * Process data with various operations.
 *
 * This class demonstrates typical PHP class structure with
 * methods, properties, and PHPDoc documentation.
 */
class DataProcessor {
    /**
     * @var string The processor name
     */
    private $name;

    /**
     * @var array Internal cache
     */
    private $cache;

    /**
     * Initialize the processor.
     *
     * @param string $name The name of this processor instance
     */
    public function __construct($name) {
        $this->name = $name;
        $this->cache = array();
    }

    /**
     * Process input data.
     *
     * @param string $data The data to process
     * @return string The processed data
     */
    public function process($data) {
        return strtoupper($data);
    }

    /**
     * Process data with caching.
     *
     * @param string $data The data to process
     * @return string Processed data from cache or newly processed
     */
    public function cachedProcess($data) {
        if (!isset($this->cache[$data])) {
            $this->cache[$data] = $this->process($data);
        }
        return $this->cache[$data];
    }

    /**
     * Get the current cache size.
     *
     * @return int Number of items in cache
     */
    public function getCacheSize() {
        return count($this->cache);
    }

    /**
     * Clear the internal cache.
     *
     * This is a private helper method.
     */
    private function clearCache() {
        $this->cache = array();
    }
}

/**
 * An advanced processor with additional capabilities.
 */
class AdvancedProcessor extends DataProcessor {
    /**
     * Process data with advanced techniques.
     *
     * @param string $data The data to process
     * @return string Advanced processed data
     */
    public function process($data) {
        $result = parent::process($data);
        return "[ADVANCED] " . $result;
    }
}
