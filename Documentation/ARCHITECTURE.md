# Runway Boutique - Project Structure

## Overview
The project has been refactored using **Modular Organization** with a **hybrid ML + LLM approach** for better maintainability, scalability, and accuracy.

## Project Structure

```
streamlit-ai-text-analyzer-demo/
├── RunwayML.py                          # Main UI application (Streamlit)
├── config.py                            # Configuration and constants
├── styles.csv                           # Dataset
├── requirements.txt                     # Python dependencies
│
├── services/                            # Business logic & ML
│   ├── __init__.py
│   ├── data_service.py                  # Data loading, cleaning, validation
│   ├── ollama_service.py                # LLM API integration (Ollama)
│   └── ml_classifier_service.py         # ⭐ Random Forest ML classifier
│
├── utils/                               # Utility functions
│   ├── __init__.py
│   └── prompts.py                       # Prompt templates
│
├── models/                              # ⭐ Trained ML models
│   └── product_classifier.pkl           # Saved Random Forest model
│
├── .vscode/
│   └── launch.json                      # VS Code debug configuration
│
├── train_model.py                       # ⭐ Script to train ML model
├── test_ml_classifier.py                # ⭐ Script to test predictions
├── ML_DOCUMENTATION.md                  # ⭐ ML classifier documentation
├── ARCHITECTURE.md                      # This file
│
└── run scripts
    ├── run_windows.ps1
    ├── run_macos.sh
    └── run.sh
```

## Processing Pipeline

### Before (LLM-only)
```
Product Data → Ollama LLM → Text Analysis
```

### After (ML + LLM Hybrid) ⭐
```
Product Data
    ↓
Data Cleaning & Validation (data_service.py)
    ↓
Random Forest ML Classifier (ml_classifier_service.py)
    ├─→ Prediction: Which collection? (Rainy/Sunny/Others)
    └─→ Confidence Score: How certain? (0-100%)
    ↓
Ollama LLM (ollama_service.py)
    └─→ Detailed Explanation & Recommendations
    ↓
UI Display: Prediction + Confidence + Explanation
```

## File Descriptions

### Core Application
- **RunwayML.py**: Main Streamlit UI entry point. Orchestrates data, ML, and LLM.
- **config.py**: Centralized configuration (Ollama URL, model names, display settings).

### Services (Business Logic)
- **services/data_service.py**: 
  - Loads CSV data with caching
  - Cleans data (removes duplicates, NaN values)
  - Validates dataset integrity
  - Extracts and formats product information

- **services/ollama_service.py**: 
  - Integrates with Ollama LLM service
  - Makes API calls for text generation
  - Handles connection errors gracefully
  - Provides singleton instance for reuse

- **services/ml_classifier_service.py**: ⭐ **NEW**
  - Random Forest classifier for product recategorization
  - Maps seasons to weather collections:
    - Winter/Fall → "Rainy Days Collection"
    - Spring/Summer → "Sunny Days Collection"
    - Others → "Others"
  - Provides single and batch predictions
  - Calculates confidence scores
  - Saves/loads trained models
  - Features used: masterCategory, subCategory, articleType, baseColour, usage

### Utilities
- **utils/prompts.py**: Task definitions and prompt templates for Ollama.

### Training & Testing
- **train_model.py**: ⭐ **NEW** - Trains Random Forest on dataset
  - Cleans and validates data
  - Splits into train/test sets
  - Displays accuracy, confusion matrix, feature importance
  - Saves model to `models/product_classifier.pkl`
  - Run: `python train_model.py`

- **test_ml_classifier.py**: ⭐ **NEW** - Tests ML classifier functionality
  - Single product prediction
  - Batch prediction on multiple products
  - Season-to-collection mapping
  - Feature importance visualization
  - Run: `python test_ml_classifier.py`

### Documentation
- **ML_DOCUMENTATION.md**: ⭐ **NEW** - Complete ML classifier guide with examples
- **ARCHITECTURE.md**: This file - project structure and design overview

## ML Classification Details

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Trees**: 100 decision trees
- **Max Depth**: 15 levels
- **Train/Test Split**: 80% training, 20% testing
- **Typical Accuracy**: 85-90%

### Features Used
1. `masterCategory` - Product category (Apparel, Footwear, etc.)
2. `subCategory` - Specific subcategory
3. `articleType` - Type of item (Sweater, Boots, T-shirt, etc.)
4. `baseColour` - Primary color
5. `usage` - Usage context (Casual, Formal, Sports, etc.)

### Output Classes
- **Rainy Days Collection**: Winter/Fall products
- **Sunny Days Collection**: Spring/Summer products
- **Others**: Off-season or unsuitable items

## How to Run

### Option 1: Terminal (Production)
```powershell
.\run_windows.ps1  # Windows
./run_macos.sh     # macOS
./run.sh           # Linux
```

### Option 2: VS Code Debug (Development) 
Press **F5** or Run → Start Debugging
- Enables breakpoints and stepping
- Full debug information

### Option 3: Direct Command
```bash
streamlit run RunwayML.py
```

---

## Setting Up ML Classifier

### First Time Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model** (one-time):
   ```bash
   python train_model.py
   ```
   Output: `models/product_classifier.pkl` with trained model

3. **Test the model** (optional):
   ```bash
   python test_ml_classifier.py
   ```
   Verifies predictions work correctly

4. **Run the app**:
   ```bash
   streamlit run RunwayML.py
   ```

### Using the ML Classifier in Code

```python
from services.ml_classifier_service import product_classifier

# Single prediction
result = product_classifier.predict({
    'masterCategory': 'Apparel',
    'subCategory': 'Topwear',
    'articleType': 'Sweater',
    'baseColour': 'Navy',
    'usage': 'Casual'
})
print(f"Collection: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
df_results = product_classifier.predict_batch(df)
```

## Key Improvements

✅ **Proper ML Pipeline**: Random Forest classifier with feature engineering
✅ **Data Cleaning**: Automatic removal of duplicates and NaN values
✅ **Hybrid Approach**: ML for accuracy + LLM for explanation
✅ **Reusable Models**: Save/load trained models for quick startup
✅ **Modular Design**: Each component has single responsibility
✅ **Type Hints**: Better IDE support and error detection
✅ **Comprehensive Logging**: Debug-friendly with detailed logs
✅ **Testable**: Each service can be tested independently
✅ **Configurable**: All settings in `config.py`

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.41.1 | Web UI framework |
| pandas | 2.1.3 | Data processing |
| scikit-learn | 1.3.2 | ML algorithms (Random Forest) |
| numpy | 1.24.3 | Numerical computing |
| requests | 2.32.3 | HTTP client for Ollama |

## Next Steps to Enhance

1. ✅ ML Classifier (DONE)
2. ⏳ **UI Integration**: Show ML predictions in RunwayML.py
3. ⏳ **Auto-Training**: Retrain model periodically
4. ⏳ **Advanced Models**: Try XGBoost, LightGBM
5. ⏳ **Feature Engineering**: Add price, rating, availability
6. ⏳ **Model Explainability**: Use SHAP for prediction explanations
7. ⏳ **A/B Testing**: Compare ML vs LLM predictions
8. ⏳ **Performance Monitoring**: Track model accuracy over time

- **services/data_service.py**: 
  - Loads and caches CSV data
  - Provides product selection/filtering utilities
  - Formats data for display

### Utilities
- **utils/prompts.py**: 
  - Stores all prompt templates
  - Generates system prompts for Ollama
  - Easy to extend with new tasks

## How to Run

### Option 1: Terminal (Traditional)
```powershell
.\run_windows.ps1
```

### Option 2: VS Code Debug Mode (Recommended for Development)
1. Press **F5** or go to **Run → Start Debugging**
2. Select "Streamlit RunwayML" configuration
3. Streamlit app will start with debug capabilities (breakpoints, etc.)

### Option 3: Direct Command
```bash
streamlit run RunwayML.py
```

## Key Improvements

**Separation of Concerns**
- UI logic in `RunwayML.py`
- ML logic in `services/ollama_service.py`
- Data handling in `services/data_service.py`
- Configuration centralized in `config.py`

**Reusability**
- Services can be used in other projects
- Prompts can be easily modified or extended
- Configuration is centralized

**Testability**
- Each module can be tested independently
- Services have clear interfaces

**Maintainability**
- Easier to find and modify code
- Clear responsibility for each module
- Type hints for better IDE support

**Debugging**
- VS Code debug configuration included
- Logging support for troubleshooting

## Adding New Features

### Add a New AI Task
1. Edit `utils/prompts.py`
2. Add entry to `TASK_PROMPTS` dictionary

Example:
```python
"New Task": "Instructions for the new task..."
```

### Extend Data Service
Add new functions to `services/data_service.py` for data processing:
```python
def get_product_by_category(df, category):
    return df[df['masterCategory'] == category]
```

### Modify Configuration
Edit `config.py` to change any setting:
```python
OLLAMA_URL = "your_new_url"
DEFAULT_MODEL = "new_model"
```

## Dependencies
See `requirements.txt` for all dependencies. Key ones:
- `streamlit` - UI framework
- `pandas` - Data processing
- `requests` - HTTP client for Ollama API

## Troubleshooting

**"Cannot connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check that `OLLAMA_URL` in `config.py` is correct

**"styles.csv not found"**
- Ensure `styles.csv` is in the same directory as `RunwayML.py`

**Debug breakpoints not working**
- Make sure you're running via F5 (debug mode) not terminal
- Check `.vscode/launch.json` exists
