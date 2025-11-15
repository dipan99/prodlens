"""
Script to extract unique brands from product CSV files and create brands.csv
"""

import pandas as pd
import os

def create_brands_csv():
    """Extract unique brands from monitors, mice, and keyboards CSV files."""
    
    print("Reading product CSV files...")
    
    # Read all three files
    monitors = pd.read_csv('raw_data/monitors_clean2.csv')
    mice = pd.read_csv('raw_data/mice_clean2.csv')
    keyboards = pd.read_csv('raw_data/keyboards_clean2.csv')
    
    print(f"Monitors: {len(monitors)} rows")
    print(f"Mice: {len(mice)} rows")
    print(f"Keyboards: {len(keyboards)} rows")
    
    # Extract Brand columns
    monitor_brands = monitors[['Brand']].copy()
    mice_brands = mice[['Brand']].copy()
    keyboard_brands = keyboards[['Brand']].copy()
    
    # Combine all brands
    all_brands = pd.concat([monitor_brands, mice_brands, keyboard_brands])
    
    # Get unique brands
    unique_brands = all_brands.drop_duplicates().dropna()
    
    print(f"\nTotal unique brands found: {len(unique_brands)}")
    
    # Rename column to match database schema
    unique_brands.columns = ['brand_name']
    
    # Add empty columns for country_origin and website_url
    # These will be filled later using Perplexity API or manually
    unique_brands['country_origin'] = None
    unique_brands['website_url'] = None
    
    # Sort by brand name
    unique_brands = unique_brands.sort_values('brand_name').reset_index(drop=True)
    
    # Save to CSV
    output_path = 'raw_data/brands.csv'
    unique_brands.to_csv(output_path, index=False)
    
    print(f"\nBrands CSV created: {output_path}")
    print("\nFirst 10 brands:")
    print(unique_brands.head(10))
    
    print("\nLast 10 brands:")
    print(unique_brands.tail(10))
    
    return unique_brands

if __name__ == "__main__":
    print("=" * 60)
    print("CREATING BRANDS CSV FILE")
    print("=" * 60)
    
    brands_df = create_brands_csv()
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! {len(brands_df)} unique brands extracted")
    print("=" * 60)