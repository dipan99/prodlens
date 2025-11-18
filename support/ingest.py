from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from logger import Logging
import pandas as pd
import numpy as np
import psycopg2
import os
import re

Logging.setLevel()
load_dotenv()


class ElectronicsDataPipeline:
    """
    Complete ingestion pipeline for electronics data from CSV files.
    Handles brands, products, and all product specifications.
    """
    
    def __init__(self, db_config: Dict[str, str]) -> None:
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.brand_cache = {}  # Cache brand_id lookups
    

    def connect(self) -> None:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            Logging.logInfo("Database connection established")
        except Exception as e:
            Logging.logError(str(e))
            raise e
    

    def close(self) -> None:
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            Logging.logInfo("Database connection closed")
        except Exception as e:
            Logging.logError(str(e))
            raise
    

    def clean_value(self, value, expected_type=str):
        """
        Clean and convert value to expected type.
        Handles NaN, empty strings, 'inf', 'No', and various formats.
        """
        try:
            if value is None or pd.isna(value) or value == '':
                return None

            s = str(value).strip()

            # Handle 'inf' specially
            if 'inf' in s.lower():
                return None

            # ---------- CASE 1: expected TYPE is string ----------
            if expected_type == str:
                return s

            # ---------- CASE 2: expected TYPE is integer ----------
            elif expected_type == int:
                # Handle text values that mean "0" or "None"
                if s.lower() in ['no', 'none', 'n/a', 'na']:
                    return 0  # or return None if you prefer NULL in database
                
                # If value contains "/", take ONLY first part
                if "/" in s:
                    s = s.split("/")[0].strip()

                # Extract first number (integer or decimal)
                match = re.search(r"\d+(\.\d+)?", s)
                if not match:
                    return None

                num = float(match.group(0))  # may be decimal
                return int(round(num))       # cast to int safely

            # ---------- CASE 3: expected TYPE is float ----------
            elif expected_type == float:
                # Handle text values
                if s.lower() in ['no', 'none', 'n/a', 'na']:
                    return None
                
                if "/" in s:
                    s = s.split("/")[0].strip()

                match = re.search(r"\d+(\.\d+)?", s)
                if not match:
                    return None
                
                return float(match.group(0))

            # ---------- CASE 4: expected TYPE is boolean ----------
            elif expected_type == bool:
                if isinstance(value, bool):
                    return value
                
                value_str = str(value).strip().lower()
                if value_str in ['yes', 'true', '1', 't']:
                    return True
                elif value_str in ['no', 'false', '0', 'f']:
                    return False
                else:
                    return None

            return None
        except Exception as e:
            Logging.logError(str(e))
            raise e
            

    def ingest_brands(self, df: pd.DataFrame):
        try:
            ingested = 0
            skipped = 0
            
            for idx, row in df.iterrows():
                try:
                    brand_name = self.clean_value(row.get('brand_name'))
                    
                    if not brand_name:
                        skipped += 1
                        continue
                    
                    # Check if brand exists
                    self.cursor.execute(
                        "SELECT brand_id FROM brands WHERE brand_name = %s",
                        (brand_name,)
                    )
                    if self.cursor.fetchone():
                        Logging.logDebug(f"Brand '{brand_name}' already exists, skipping")
                        skipped += 1
                        continue
                    
                    # Insert brand
                    self.cursor.execute(
                        """
                        INSERT INTO brands (brand_name, country_origin, website_url)
                        VALUES (%s, %s, %s)
                        RETURNING brand_id
                        """,
                        (
                            brand_name,
                            self.clean_value(row.get('country_origin')),
                            self.clean_value(row.get('website_url'))
                        )
                    )
                    brand_id = self.cursor.fetchone()['brand_id']
                    self.conn.commit()
                    
                    self.brand_cache[brand_name] = brand_id
                    ingested += 1
                    
                except Exception as e:
                    self.conn.rollback()
                    Logging.logError(f"Error ingesting brand at row {idx}: {e}")
                    continue
            
            Logging.logInfo(f"Brands ingestion complete: {ingested} ingested, {skipped} skipped")
            
        except Exception as e:
            Logging.logError(f"Error in brands ingestion: {e}")
            raise


    def get_brand_id(self, brand_name: str) -> Optional[int]:
        if not brand_name:
            return None
        
        # Check cache
        if brand_name in self.brand_cache:
            return self.brand_cache[brand_name]
        
        # Query database
        self.cursor.execute(
            "SELECT brand_id FROM brands WHERE brand_name = %s",
            (brand_name,)
        )
        result = self.cursor.fetchone()
        
        if result:
            self.brand_cache[brand_name] = result['brand_id']
            return result['brand_id']
        
        Logging.logWarning(f"Brand '{brand_name}' not found in database")
        return None


    def product_exists(self, product_name: str, brand_id: int, category: str) -> Optional[int]:
        """
        Check if product already exists.
        
        Returns:
            product_id if exists, None otherwise
        """
        try:
            self.cursor.execute(
                """
                SELECT product_id FROM products 
                WHERE product_name = %s AND brand_id = %s AND category_name = %s
                """,
                (product_name, brand_id, category)
            )
            result = self.cursor.fetchone()
            return result['product_id'] if result else None
        except Exception as e:
            Logging.logError(str(e))
            raise e

    
    def ingest_monitors(self, df: pd.DataFrame):
        try:
            ingested = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    product_name = self.clean_value(row.get('Product'))
                    brand_name = self.clean_value(row.get('Brand'))
                    
                    if not product_name or not brand_name:
                        Logging.logWarning(f"Row {idx}: Missing product name or brand")
                        failed += 1
                        continue
                    
                    brand_id = self.get_brand_id(brand_name)
                    if not brand_id:
                        Logging.logWarning(f"Row {idx}: Brand '{brand_name}' not found")
                        failed += 1
                        continue
                    
                    # Insert product
                    self.cursor.execute(
                        """
                        INSERT INTO products (
                            product_name, brand_id, category_name, release_year, price, product_link
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING product_id
                        """,
                        (
                            product_name,
                            brand_id,
                            'Monitor',
                            self.clean_value(row.get('Release Year'), int),
                            None, #self.clean_value(row.get('Price')),
                            None  # No product_link in CSV
                        )
                    )
                    product_id = self.cursor.fetchone()['product_id']
                    
                    # Insert monitor specs
                    self.cursor.execute(
                        """
                        INSERT INTO monitor_specs (
                            product_id, size_inch, curve_radius, wall_mount, borders_size_cm,
                            brightness_rating, response_time_rating, hdr_picture_rating,
                            sdr_picture_rating, color_accuracy_rating,
                            pixel_type, subpixel_layout, backlight, color_depth_bit,
                            native_contrast, contrast_with_local_dimming, local_dimming,
                            sdr_real_scene_cdm2, sdr_peak_100_window_cdm2, sdr_sustained_100_window_cdm2,
                            hdr_real_scene_cdm2, hdr_peak_100_window_cdm2, hdr_sustained_100_window_cdm2,
                            minimum_brightness_cdm2, white_balance_dE, black_uniformity_native_std_dev,
                            color_washout_from_left_degrees, color_washout_from_right_degrees,
                            color_shift_from_left_degrees, color_shift_from_right_degrees,
                            brightness_loss_from_left_degrees, brightness_loss_from_right_degrees,
                            black_level_raise_from_left_degrees, black_level_raise_from_right_degrees,
                            native_refresh_rate_hz, max_refresh_rate_hz, native_resolution,
                            aspect_ratio, flicker_free,
                            max_refresh_rate_over_hdmi_hz, displayport, hdmi, usbc_ports
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s
                        )
                        """,
                        (
                            product_id,
                            self.clean_value(row.get('Size (inch)'), float),
                            self.clean_value(row.get('Curve Radius'), str),
                            self.clean_value(row.get('Wall Mount'), str),
                            self.clean_value(row.get('Borders Size (cm)'), float),
                            self.clean_value(row.get('Brightness'), float),
                            self.clean_value(row.get('Response Time'), float),
                            self.clean_value(row.get('HDR Picture'), float),
                            self.clean_value(row.get('SDR Picture'), float),
                            self.clean_value(row.get('Color Accuracy'), float),
                            self.clean_value(row.get('Pixel Type'), str),
                            self.clean_value(row.get('Subpixel Layout'), str),
                            self.clean_value(row.get('Backlight'), str),
                            self.clean_value(row.get('Color Depth (Bit)'), int),
                            self.clean_value(row.get('Native Contrast'), float),
                            self.clean_value(row.get('Contrast With Local Dimming'), float),
                            self.clean_value(row.get('Local Dimming'), bool),
                            self.clean_value(row.get('SDR Real Scene (cd/m2)'), float),
                            self.clean_value(row.get('SDR Peak 100% Window (cd/m2)'), float),
                            self.clean_value(row.get('SDR Sustained 100% Window (cd/m2)'), float),
                            self.clean_value(row.get('HDR Real Scene (cd/m2)'), float),
                            self.clean_value(row.get('HDR Peak 100% Window (cd/m2)'), float),
                            self.clean_value(row.get('HDR Sustained 100% Window (cd/m2)'), float),
                            self.clean_value(row.get('Minimum Brightness (cd/m2)'), float),
                            self.clean_value(row.get('White Balance (dE)'), float),
                            self.clean_value(row.get('Black Uniformity Native (Std Dev)'), float),
                            self.clean_value(row.get('Color Washout From Left (degrees)'), int),
                            self.clean_value(row.get('Color Washout From Right (degrees)'), int),
                            self.clean_value(row.get('Color Shift From Left (degrees)'), int),
                            self.clean_value(row.get('Color Shift From Right (degrees)'), int),
                            self.clean_value(row.get('Brightness Loss From Left (degrees)'), int),
                            self.clean_value(row.get('Brightness Loss From Right (degrees)'), int),
                            self.clean_value(row.get('Black Level Raise From Left (degrees)'), int),
                            self.clean_value(row.get('Black Level Raise From Right (degrees)'), int),
                            self.clean_value(row.get('Native Refresh Rate (Hz)'), int),
                            self.clean_value(row.get('Max Refresh Rate (Hz)'), int),
                            self.clean_value(row.get('Native Resolution'), str),
                            self.clean_value(row.get('Aspect Ratio'), str),
                            self.clean_value(row.get('Flicker-Free'), bool),
                            self.clean_value(row.get('Max Refresh Rate Over HDMI (Hz)'), int),
                            self.clean_value(row.get('DisplayPort'), str),
                            self.clean_value(row.get('HDMI'), str),
                            self.clean_value(row.get('USB-C Ports'), int),
                        )
                    )
                    
                    # Insert professional ratings
                    product_slug = product_name.lower()\
                        .replace(brand_name.lower(), '').strip()\
                        .replace(' ', '-').replace('--', '-').strip('-')
                    self.cursor.execute(
                        """
                        INSERT INTO professional_ratings (
                            product_id, reviewer_website,
                            rating_general, rating_gaming, rating_office, 
                            rating_editing, review_url
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            product_id,
                            'https://www.rtings.com',
                            self.clean_value(row.get('Ranking General'), float),
                            self.clean_value(row.get('Ranking Gaming'), float),
                            self.clean_value(row.get('Ranking Office'), float),
                            self.clean_value(row.get('Ranking Editing'), float),
                            f'https://www.rtings.com/monitor/reviews/{brand_name.lower()}/{product_slug}'
                        )
                    )
                    
                    self.conn.commit()
                    ingested += 1
                    
                    if ingested % 10 == 0:
                        Logging.logInfo(f"Progress: {ingested} monitors ingested")
                    
                except Exception as e:
                    self.conn.rollback()
                    Logging.logError(f"Error ingesting monitor at row {idx}: {e}")
                    failed += 1
                    continue
            
            Logging.logInfo(f"Monitors ingestion complete: {ingested} ingested, {failed} failed")
            
        except Exception as e:
            Logging.logError(str(e))
            raise e

    
    def ingest_mice(self, df: pd.DataFrame):
        try:            
            ingested = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    product_name = self.clean_value(row.get('Product'))
                    brand_name = self.clean_value(row.get('Brand'))
                    
                    if not product_name or not brand_name:
                        Logging.logWarning(f"Row {idx}: Missing product name or brand")
                        failed += 1
                        continue
                    
                    brand_id = self.get_brand_id(brand_name)
                    if not brand_id:
                        Logging.logWarning(f"Row {idx}: Brand '{brand_name}' not found")
                        failed += 1
                        continue
                    
                    # Insert product
                    self.cursor.execute(
                        """
                        INSERT INTO products (
                            product_name, brand_id, category_name, release_year
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING product_id
                        """,
                        (
                            product_name,
                            brand_id,
                            'Mouse',
                            self.clean_value(row.get('Release Year'))
                        )
                    )
                    product_id = self.cursor.fetchone()['product_id']
                    
                    # Insert mouse specs
                    self.cursor.execute(
                        """
                        INSERT INTO mouse_specs (
                            product_id, coating, length_mm, width_mm, height_mm, grip_width_mm,
                            default_weight_gm, weight_distribution, ambidextrous,
                            left_handed_friendly, finger_rest,
                            total_number_of_buttons, number_of_side_buttons,
                            profile_switching_button, scroll_wheel_type,
                            connectivity, battery_type, maximum_of_paired_devices, cable_length_m,
                            mouse_feet_material, switch_type, switch_model,
                            software_windows_compatibility, software_macos_compatibility
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """,
                        (
                            product_id,
                            self.clean_value(row.get('Coating'), str),
                            self.clean_value(row.get('Length (mm)'), float),
                            self.clean_value(row.get('Width (mm)'), float),
                            self.clean_value(row.get('Height (mm)'), float),
                            self.clean_value(row.get('Grip Width (mm)'), float),
                            self.clean_value(row.get('Default Weight (gm)'), float),
                            self.clean_value(row.get('Weight Distribution'), str),
                            self.clean_value(row.get('Ambidextrous'), str),
                            self.clean_value(row.get('Left-Handed Friendly'), bool),     # BOOLEAN
                            self.clean_value(row.get('Finger Rest'), bool),              # BOOLEAN
                            self.clean_value(row.get('Total Number Of Buttons'), int),
                            self.clean_value(row.get('Number Of Side Buttons'), int),
                            self.clean_value(row.get('Profile Switching Button'), bool),  # BOOLEAN
                            self.clean_value(row.get('Scroll Wheel Type'), str),
                            self.clean_value(row.get('Connectivity'), str),
                            self.clean_value(row.get('Battery Type'), str),
                            self.clean_value(row.get('Maximum Of Paired Devices'), str),
                            self.clean_value(row.get('Cable Length (m)'), float),
                            self.clean_value(row.get('Mouse Feet Material'), str),
                            self.clean_value(row.get('Switch Type'), str),
                            self.clean_value(row.get('Switch Model'), str),
                            self.clean_value(row.get('Software Windows Compatibility'), bool), # BOOLEAN
                            self.clean_value(row.get('Software macOS Compatibility'), bool)    # BOOLEAN

                        )
                    )
                    
                    # Insert professional ratings
                    product_slug = product_name.lower()\
                        .replace(brand_name.lower(), '').strip()\
                        .replace(' ', '-').replace('--', '-').strip('-')
                    self.cursor.execute(
                        """
                        INSERT INTO professional_ratings (
                            product_id, reviewer_website,
                            rating_general, rating_gaming, rating_office, 
                            rating_editing, review_url
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            product_id,
                            'https://www.rtings.com',
                            self.clean_value(row.get('Ranking General'), float),
                            self.clean_value(row.get('Ranking Gaming'), float),
                            self.clean_value(row.get('Ranking Office'), float),
                            self.clean_value(row.get('Ranking Editing'), float),
                            f'https://www.rtings.com/mouse/reviews/{brand_name.lower()}/{product_slug}'

                        )
                    )
                    
                    self.conn.commit()
                    ingested += 1
                    
                    if ingested % 10 == 0:
                        Logging.logInfo(f"Progress: {ingested} mice ingested")
                    
                except Exception as e:
                    self.conn.rollback()
                    Logging.logError(f"Error ingesting mouse at row {idx}: {e}")
                    failed += 1
                    continue
            
            Logging.logInfo(f"Mice ingestion complete: {ingested} ingested, {failed} failed")
            
        except Exception as e:
            Logging.logError(str(e))
            raise e

    
    def ingest_keyboards(self, df: pd.DataFrame):
        try:
            ingested = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    product_name = self.clean_value(row.get('Product'))
                    brand_name = self.clean_value(row.get('Brand'))
                    
                    if not product_name or not brand_name:
                        Logging.logWarning(f"Row {idx}: Missing product name or brand")
                        failed += 1
                        continue
                    
                    brand_id = self.get_brand_id(brand_name)
                    if not brand_id:
                        Logging.logWarning(f"Row {idx}: Brand '{brand_name}' not found")
                        failed += 1
                        continue
                    
                    # Insert product
                    self.cursor.execute(
                        """
                        INSERT INTO products (
                            product_name, brand_id, category_name, release_year
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING product_id
                        """,
                        (
                            product_name,
                            brand_id,
                            'Keyboard',
                            self.clean_value(row.get('Release Year'), int)
                        )
                    )
                    product_id = self.cursor.fetchone()['product_id']
                    
                    # Insert keyboard specs
                    self.cursor.execute(
                        """
                        INSERT INTO keyboard_specs (
                            product_id, size, height_cm, width_cm, depth_cm,
                            depth_with_wrist_rest_cm, weight_kg,
                            keycap_material, curved_or_angled, split_keyboard,
                            replaceable_cherry_stabilizers, switch_stem_shape,
                            mechanical_switch_compatibility, magnetic_switch_compatibility,
                            backlighting, rgb, per_key_backlighting, effects,
                            connectivity, detachable, connector_length_m, connector_keyboard_side, bluetooth,
                            media_keys, trackpad_or_trackball, scroll_wheel, numpad, windows_key_lock,
                            key_spacing_mm, average_loudness_dba,
                            pre_travel_mm, total_travel_mm, detection_ratio_percent,
                            switch_type, switch_feel, software_configuration_profiles,
                            windows_compatibility, macos_compatibility, linux_compatibility
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            product_id,
                            self.clean_value(row.get('Size'), str),
                            self.clean_value(row.get('Height (cm)'), float),
                            self.clean_value(row.get('Width (cm)'), float),
                            self.clean_value(row.get('Depth (cm)'), float),
                            self.clean_value(row.get('Depth With Wrist Rest (cm)'), float),
                            self.clean_value(row.get('Weight (kg)'), float),
                            self.clean_value(row.get('Keycap Material'), str),
                            self.clean_value(row.get('Curved or Angled'), bool),         
                            self.clean_value(row.get('Split Keyboard'), bool),           
                            self.clean_value(row.get('Replaceable Cherry Stabilizers'), bool), 
                            self.clean_value(row.get('Switch Stem Shape'), str),
                            self.clean_value(row.get('Mechanical Switch Compatibility'), str),
                            self.clean_value(row.get('Magnetic Switch Compatibility'), str),
                            self.clean_value(row.get('Backlighting'), bool),                  
                            self.clean_value(row.get('RGB'), bool),                           
                            self.clean_value(row.get('Per-Key Backlighting'), bool),          
                            self.clean_value(row.get('Effects'), bool),                       
                            self.clean_value(row.get('Connectivity'), str),
                            self.clean_value(row.get('Detachable'), str),
                            self.clean_value(row.get('Connector Length (m)'), float),
                            self.clean_value(row.get('Connector (Keyboard side)'), str),
                            self.clean_value(row.get('Bluetooth'), bool),                     
                            self.clean_value(row.get('Media Keys'), str),
                            self.clean_value(row.get('Trackpad or Trackball'), bool),         
                            self.clean_value(row.get('Scroll Wheel'), bool),                  
                            self.clean_value(row.get('Numpad'), bool),                        
                            self.clean_value(row.get('Windows Key Lock'), bool),              
                            self.clean_value(row.get('Key Spacing (mm)'), float),
                            self.clean_value(row.get('Average Loudness (dBA)'), float),
                            self.clean_value(row.get('Pre-Travel (mm)'), float),
                            self.clean_value(row.get('Total Travel (mm)'), float),
                            self.clean_value(row.get('Detection Ratio (%)'), float),
                            self.clean_value(row.get('Switch Type'), str),
                            self.clean_value(row.get('Switch Feel'), str), 
                            self.clean_value(row.get('Software Configuration Profiles'), str),
                            self.clean_value(row.get('Windows'), str),
                            self.clean_value(row.get('macOS'), str),
                            self.clean_value(row.get('Linux'), str)
                        )
                    )
                    
                    # Insert professional ratings
                    product_slug = product_name.lower()\
                        .replace(brand_name.lower(), '').strip()\
                        .replace(' ', '-').replace('--', '-').strip('-')
                    self.cursor.execute(
                        """
                        INSERT INTO professional_ratings (
                            product_id, reviewer_website,
                            rating_general, rating_gaming, rating_office, 
                            rating_editing, review_url
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            product_id,
                            'https://www.rtings.com',
                            self.clean_value(row.get('Ranking General'), float),
                            self.clean_value(row.get('Ranking Gaming'), float),
                            self.clean_value(row.get('Ranking Office'), float),
                            self.clean_value(row.get('Ranking Editing'), float),
                            f'https://www.rtings.com/keyboard/reviews/{brand_name.lower()}/{product_slug}'
                        )
                    )
                    
                    self.conn.commit()
                    ingested += 1
                    
                    if ingested % 10 == 0:
                        Logging.logInfo(f"Progress: {ingested} keyboards ingested")
                    
                except Exception as e:
                    self.conn.rollback()
                    Logging.logError(f"Error ingesting keyboard at row {idx}: {e}")
                    failed += 1
                    continue
            
            Logging.logInfo(f"Keyboards ingestion complete: {ingested} ingested, {failed} failed")
            
        except Exception as e:
            Logging.logError(str(e))
            raise e
    

    def print_summary(self):
        """Print summary statistics of ingested data."""
        try:
            Logging.logInfo("=" * 60)
            Logging.logInfo("DATABASE SUMMARY")
            Logging.logInfo("=" * 60)
            
            # Count brands
            self.cursor.execute("SELECT COUNT(*) as count FROM brands")
            brands_count = self.cursor.fetchone()['count']
            Logging.logInfo(f"Total Brands: {brands_count}")
            
            # Count products by category
            self.cursor.execute("""
                SELECT category_name, COUNT(*) as count
                FROM products
                GROUP BY category_name
                ORDER BY category_name
            """)
            for row in self.cursor.fetchall():
                Logging.logInfo(f"Total {row['category_name']}s: {row['count']}")
            
            # Count total products
            self.cursor.execute("SELECT COUNT(*) as count FROM products")
            products_count = self.cursor.fetchone()['count']
            Logging.logInfo(f"Total Products: {products_count}")
            
            # Count ratings
            self.cursor.execute("SELECT COUNT(*) as count FROM professional_ratings")
            ratings_count = self.cursor.fetchone()['count']
            Logging.logInfo(f"Total Professional Ratings: {ratings_count}")
        except Exception as e:
            raise e


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING FULL INGESTION PIPELINE")
    print("=" * 60)
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }
    db_instance = ElectronicsDataPipeline(db_config)
    db_instance.connect()
    
    # Step 1: Ingest brands first (required for foreign keys)
    db_instance.ingest_brands(pd.read_csv(os.path.join("data", "brands.csv")))
    
    # Step 2: Ingest monitors
    db_instance.ingest_monitors(pd.read_csv(os.path.join("data", "monitors_clean2.csv"), nrows=10))

    # Step 3: Ingest mice
    db_instance.ingest_mice(pd.read_csv(os.path.join("data", "mice_clean2.csv"), nrows=10))
    
    # Step 4: Ingest keyboards
    db_instance.ingest_keyboards(pd.read_csv(os.path.join("data", "keyboards_clean2.csv"), nrows=10))
    
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    # Print summary statistics
    db_instance.print_summary()
    db_instance.close()