"""
Configuration settings for Runway Boutique AI Analyzer
"""

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:1.5b"
OLLAMA_TIMEOUT = 180

# Data Configuration
DATA_FILE = "styles.csv"
MAX_PRODUCTS_IN_DROPDOWN = 500

# UI Configuration
PAGE_TITLE = "Runway Boutique - AI Inventory Analyzer"
PAGE_ICON = "👗"
LAYOUT = "wide"

# Display Configuration
SAMPLE_ROWS_DISPLAY = 100
PRODUCT_INFO_HEIGHT = 180
