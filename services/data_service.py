"""
Data Service - Handles data loading and processing
"""
import pandas as pd
import logging
import streamlit as st
from typing import Optional
from config import DATA_FILE, MAX_PRODUCTS_IN_DROPDOWN

logger = logging.getLogger(__name__)


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and cache the styles dataset
    
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If styles.csv not found
    """
    try:
        df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} records from {DATA_FILE}")
        return df
    except FileNotFoundError:
        logger.error(f"File '{DATA_FILE}' not found")
        raise


def get_product_options(df: pd.DataFrame, max_products: int = MAX_PRODUCTS_IN_DROPDOWN):
    """
    Generate product options for dropdown
    
    Args:
        df: Input DataFrame
        max_products: Maximum products to include
        
    Returns:
        Series of formatted product options
    """
    sample_df = df.dropna(subset=['productDisplayName']).head(max_products)
    return sample_df['id'].astype(str) + " - " + sample_df['productDisplayName']


def get_product_details(df: pd.DataFrame, product_id: int) -> dict:
    """
    Get details for a specific product
    
    Args:
        df: Input DataFrame
        product_id: Product ID to retrieve
        
    Returns:
        Dictionary containing product information
    """
    product_data = df[df['id'] == product_id].iloc[0]
    return {
        'name': product_data['productDisplayName'],
        'master_category': product_data['masterCategory'],
        'sub_category': product_data['subCategory'],
        'article_type': product_data['articleType'],
        'base_color': product_data['baseColour'],
        'season': product_data['season'],
        'usage': product_data['usage']
    }


def format_product_details(product_details: dict) -> str:
    """
    Format product details as readable text
    
    Args:
        product_details: Dictionary of product information
        
    Returns:
        Formatted product details string
    """
    return f"""Product Details:
- Name: {product_details['name']}
- Master Category: {product_details['master_category']} > {product_details['sub_category']}
- Article Type: {product_details['article_type']}
- Base Color: {product_details['base_color']}
- Original Season: {product_details['season']}
- Usage: {product_details['usage']}
"""


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing duplicates and NaN values
    
    Args:
        df: Raw DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    
    # Remove rows with NaN values
    df_cleaned = df_cleaned.dropna()
    nulls_removed = len(df_cleaned) - (initial_rows - duplicates_removed)
    
    logger.info(f"Data cleaning complete: Removed {duplicates_removed} duplicates, {nulls_removed} rows with NaN values")
    logger.info(f"Original rows: {initial_rows}, Cleaned rows: {len(df_cleaned)}")
    
    return df_cleaned


def validate_data(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate if dataset has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    if required_columns is None:
        required_columns = ['id', 'productDisplayName', 'season', 'masterCategory', 'subCategory']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info(f"Data validation passed. Dataset has all required columns.")
    return True
