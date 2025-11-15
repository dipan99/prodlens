import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ElectronicsDataPipeline:
    """
    Complete ingestion pipeline for electronics data from CSV files.
    Handles brands, products, and all product specifications.
    """
    
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the ingestion pipeline.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.brand_cache = {}  # Cache brand_id lookups
        
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def clean_value(self, value: Any) -> Any:
        """Clean data values - convert NaN/empty to None."""
        if pd.isna(value) or value == 'nan' or value == '' or value == 'N/A':
            return None
        
        if isinstance(value, (int, float)) and (value == float('inf') or value == float('-inf')):
            return None
        
        return value
    
    def parse_boolean(self, value: Any) -> Optional[bool]:
        """Parse boolean values from various formats."""
        if pd.isna(value):
            return None
        
        if isinstance(value, bool):
            return value
        
        value_str = str(value).strip().lower()
        if value_str in ['yes', 'true', '1', 't']:
            return True
        elif value_str in ['no', 'false', '0', 'f']:
            return False
        
        return None
    
    def extract_first_number(self, value):
        """Extract first numeric value from string"""
        if pd.isna(value) or value == '':
            return None
        
        import re
        # Find first number (including decimals)
        match = re.search(r'\d+\.?\d*', str(value))
        if match:
            return float(match.group())
        return None
        
    # ============================================
    # BRANDS INGESTION
    # ============================================
    
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
                        logger.debug(f"Brand '{brand_name}' already exists, skipping")
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
                    logger.error(f"Error ingesting brand at row {idx}: {e}")
                    continue
            
            logger.info(f"Brands ingestion complete: {ingested} ingested, {skipped} skipped")
            
        except Exception as e:
            logger.error(f"Error in brands ingestion: {e}")
            raise
    
    def get_brand_id(self, brand_name: str) -> Optional[int]:
        """Get brand_id from cache or database."""
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
        
        logger.warning(f"Brand '{brand_name}' not found in database")
        return None

    def product_exists(self, product_name: str, brand_id: int, category: str) -> Optional[int]:
        """
        Check if product already exists.
        
        Returns:
            product_id if exists, None otherwise
        """
        self.cursor.execute(
            """
            SELECT product_id FROM products 
            WHERE product_name = %s AND brand_id = %s AND category_name = %s
            """,
            (product_name, brand_id, category)
        )
        result = self.cursor.fetchone()
        return result['product_id'] if result else None
    
    # ============================================
    # MONITORS INGESTION
    # ============================================
    
    def ingest_monitors(self, df: pd.DataFrame):
        try:
            ingested = 0
            skipped = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    product_name = self.clean_value(row.get('Product'))
                    brand_name = self.clean_value(row.get('Brand'))
                    
                    if not product_name or not brand_name:
                        logger.warning(f"Row {idx}: Missing product name or brand")
                        failed += 1
                        continue
                    
                    brand_id = self.get_brand_id(brand_name)
                    if not brand_id:
                        logger.warning(f"Row {idx}: Brand '{brand_name}' not found")
                        failed += 1
                        continue
                    
                    # Check if product already exists
                    existing_product_id = self.product_exists(product_name, brand_id, 'Monitor')
                    if existing_product_id:
                        logger.debug(f"Product '{product_name}' already exists, skipping")
                        skipped += 1
                        continue
                    
                    # Insert product (NO PRICE FIELD)
                    self.cursor.execute(
                        """
                        INSERT INTO products (
                            product_name, brand_id, category_name, release_year, product_link,
                            ranking_general, ranking_gaming, ranking_office, ranking_editing
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING product_id
                        """,
                        (
                            product_name,
                            brand_id,
                            'Monitor',
                            self.clean_value(row.get('Release Year')),
                            None,  # product_link
                            self.clean_value(row.get('Ranking General')),
                            self.clean_value(row.get('Ranking Gaming')),
                            self.clean_value(row.get('Ranking Office')),
                            self.clean_value(row.get('Ranking Editing'))
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
                            self.extract_first_number(row.get('Size (inch)')),
                            self.clean_value(row.get('Curve Radius')),
                            self.clean_value(row.get('Wall Mount')),
                            self.extract_first_number(row.get('Borders Size (cm)')),
                            self.extract_first_number(row.get('Brightness')),
                            self.extract_first_number(row.get('Response Time')),
                            self.extract_first_number(row.get('HDR Picture')),
                            self.extract_first_number(row.get('SDR Picture')),
                            self.extract_first_number(row.get('Color Accuracy')),
                            self.clean_value(row.get('Pixel Type')),
                            self.clean_value(row.get('Subpixel Layout')),
                            self.clean_value(row.get('Backlight')),
                            self.extract_first_number(row.get('Color Depth (Bit)')),
                            self.extract_first_number(row.get('Native Contrast')),
                            self.extract_first_number(row.get('Contrast With Local Dimming')),
                            self.parse_boolean(row.get('Local Dimming')),
                            self.extract_first_number(row.get('SDR Real Scene (cd/m2)')),
                            self.extract_first_number(row.get('SDR Peak 100% Window (cd/m2)')),
                            self.extract_first_number(row.get('SDR Sustained 100% Window (cd/m2)')),
                            self.extract_first_number(row.get('HDR Real Scene (cd/m2)')),
                            self.extract_first_number(row.get('HDR Peak 100% Window (cd/m2)')),
                            self.extract_first_number(row.get('HDR Sustained 100% Window (cd/m2)')),
                            self.extract_first_number(row.get('Minimum Brightness (cd/m2)')),
                            self.extract_first_number(row.get('White Balance (dE)')),
                            self.extract_first_number(row.get('Black Uniformity Native (Std Dev)')),
                            self.extract_first_number(row.get('Color Washout From Left (degrees)')),
                            self.extract_first_number(row.get('Color Washout From Right (degrees)')),
                            self.extract_first_number(row.get('Color Shift From Left (degrees)')),
                            self.extract_first_number(row.get('Color Shift From Right (degrees)')),
                            self.extract_first_number(row.get('Brightness Loss From Left (degrees)')),
                            self.extract_first_number(row.get('Brightness Loss From Right (degrees)')),
                            self.extract_first_number(row.get('Black Level Raise From Left (degrees)')),
                            self.extract_first_number(row.get('Black Level Raise From Right (degrees)')),
                            self.extract_first_number(row.get('Native Refresh Rate (Hz)')),
                            self.extract_first_number(row.get('Max Refresh Rate (Hz)')),
                            self.clean_value(row.get('Native Resolution')),
                            self.clean_value(row.get('Aspect Ratio')),
                            self.parse_boolean(row.get('Flicker-Free')),
                            self.extract_first_number(row.get('Max Refresh Rate Over HDMI (Hz)')),
                            self.clean_value(row.get('DisplayPort')),
                            self.clean_value(row.get('HDMI')),
                            self.extract_first_number(row.get('USB-C Ports'))
                        )
                    )
                    
                    self.conn.commit()
                    ingested += 1
                    
                    if ingested % 10 == 0:
                        logger.info(f"Progress: {ingested} monitors ingested")
                    
                except Exception as e:
                    self.conn.rollback()
                    logger.error(f"Error ingesting monitor at row {idx}: {e}")
                    failed += 1
                    continue
            
            logger.info(f"Monitors ingestion complete: {ingested} ingested, {skipped} skipped, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error in monitors ingestion: {e}")
            raise
    
    # ============================================
    # MICE INGESTION
    # ============================================
    
    def ingest_mice(self, df: pd.DataFrame):
        try:            
            ingested = 0
            skipped = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    product_name = self.clean_value(row.get('Product'))
                    brand_name = self.clean_value(row.get('Brand'))
                    
                    if not product_name or not brand_name:
                        logger.warning(f"Row {idx}: Missing product name or brand")
                        failed += 1
                        continue
                    
                    brand_id = self.get_brand_id(brand_name)
                    if not brand_id:
                        logger.warning(f"Row {idx}: Brand '{brand_name}' not found")
                        failed += 1
                        continue
                    
                    # Check if product already exists
                    existing_product_id = self.product_exists(product_name, brand_id, 'Mouse')
                    if existing_product_id:
                        logger.debug(f"Product '{product_name}' already exists, skipping")
                        skipped += 1
                        continue
                    
                    # Insert product
                    self.cursor.execute(
                        """
                        INSERT INTO products (
                            product_name, brand_id, category_name, release_year,
                            ranking_general, ranking_gaming, ranking_office, ranking_editing
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING product_id
                        """,
                        (
                            product_name,
                            brand_id,
                            'Mouse',
                            self.clean_value(row.get('Release Year')),
                            self.clean_value(row.get('Ranking General')),
                            self.clean_value(row.get('Ranking Gaming')),
                            self.clean_value(row.get('Ranking Office')),
                            self.clean_value(row.get('Ranking Editing'))
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
                            self.clean_value(row.get('Coating')),
                            self.clean_value(row.get('Length (mm)')),
                            self.clean_value(row.get('Width (mm)')),
                            self.clean_value(row.get('Height (mm)')),
                            self.clean_value(row.get('Grip Width (mm)')),
                            self.clean_value(row.get('Default Weight (gm)')),
                            self.clean_value(row.get('Weight Distribution')),
                            self.clean_value(row.get('Ambidextrous')),
                            self.parse_boolean(row.get('Left-Handed Friendly')),
                            self.parse_boolean(row.get('Finger Rest')),
                            self.clean_value(row.get('Total Number Of Buttons')),
                            self.clean_value(row.get('Number Of Side Buttons')),
                            self.parse_boolean(row.get('Profile Switching Button')),
                            self.clean_value(row.get('Scroll Wheel Type')),
                            self.clean_value(row.get('Connectivity')),
                            self.clean_value(row.get('Battery Type')),
                            self.clean_value(row.get('Maximum Of Paired Devices')),
                            self.clean_value(row.get('Cable Length (m)')),
                            self.clean_value(row.get('Mouse Feet Material')),
                            self.clean_value(row.get('Switch Type')),
                            self.clean_value(row.get('Switch Model')),
                            self.parse_boolean(row.get('Software Windows Compatibility')),
                            self.parse_boolean(row.get('Software macOS Compatibility'))
                        )
                    )
                    
                    self.conn.commit()
                    ingested += 1
                    
                    if ingested % 10 == 0:
                        logger.info(f"Progress: {ingested} mice ingested")
                    
                except Exception as e:
                    self.conn.rollback()
                    logger.error(f"Error ingesting mouse at row {idx}: {e}")
                    failed += 1
                    continue
            
            logger.info(f"Mice ingestion complete: {ingested} ingested, {skipped} skipped, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error in mice ingestion: {e}")
            raise
    
    # ============================================
    # KEYBOARDS INGESTION
    # ============================================
    
    def ingest_keyboards(self, df: pd.DataFrame):
        try:
            ingested = 0
            skipped = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    product_name = self.clean_value(row.get('Product'))
                    brand_name = self.clean_value(row.get('Brand'))
                    
                    if not product_name or not brand_name:
                        logger.warning(f"Row {idx}: Missing product name or brand")
                        failed += 1
                        continue
                    
                    brand_id = self.get_brand_id(brand_name)
                    if not brand_id:
                        logger.warning(f"Row {idx}: Brand '{brand_name}' not found")
                        failed += 1
                        continue
                    
                    # Check if product already exists
                    existing_product_id = self.product_exists(product_name, brand_id, 'Keyboard')
                    if existing_product_id:
                        logger.debug(f"Product '{product_name}' already exists, skipping")
                        skipped += 1
                        continue
                    
                    # Insert product
                    self.cursor.execute(
                        """
                        INSERT INTO products (
                            product_name, brand_id, category_name, release_year,
                            ranking_general, ranking_gaming, ranking_office, ranking_editing
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING product_id
                        """,
                        (
                            product_name,
                            brand_id,
                            'Keyboard',
                            self.clean_value(row.get('Release Year')),
                            self.clean_value(row.get('Ranking General')),
                            self.clean_value(row.get('Ranking Gaming')),
                            self.clean_value(row.get('Ranking Office')),
                            self.clean_value(row.get('Ranking Editing'))
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
                            self.clean_value(row.get('Size')),
                            self.clean_value(row.get('Height (cm)')),
                            self.clean_value(row.get('Width (cm)')),
                            self.clean_value(row.get('Depth (cm)')),
                            self.clean_value(row.get('Depth With Wrist Rest (cm)')),
                            self.clean_value(row.get('Weight (kg)')),
                            self.clean_value(row.get('Keycap Material')),
                            self.parse_boolean(row.get('Curved or Angled')),
                            self.parse_boolean(row.get('Split Keyboard')),
                            self.parse_boolean(row.get('Replaceable Cherry Stabilizers')),
                            self.clean_value(row.get('Switch Stem Shape')),
                            self.clean_value(row.get('Mechanical Switch Compatibility')),
                            self.clean_value(row.get('Magnetic Switch Compatibility')),
                            self.parse_boolean(row.get('Backlighting')),
                            self.parse_boolean(row.get('RGB')),
                            self.parse_boolean(row.get('Per-Key Backlighting')),
                            self.parse_boolean(row.get('Effects')),
                            self.clean_value(row.get('Connectivity')),
                            self.clean_value(row.get('Detachable')),
                            self.clean_value(row.get('Connector Length (m)')),
                            self.clean_value(row.get('Connector (Keyboard side)')),
                            self.parse_boolean(row.get('Bluetooth')),
                            self.clean_value(row.get('Media Keys')),
                            self.parse_boolean(row.get('Trackpad or Trackball')),
                            self.parse_boolean(row.get('Scroll Wheel')),
                            self.parse_boolean(row.get('Numpad')),
                            self.parse_boolean(row.get('Windows Key Lock')),
                            self.clean_value(row.get('Key Spacing (mm)')),
                            self.clean_value(row.get('Average Loudness (dBA)')),
                            self.clean_value(row.get('Pre-Travel (mm)')),
                            self.clean_value(row.get('Total Travel (mm)')),
                            self.clean_value(row.get('Detection Ratio (%)')),
                            self.clean_value(row.get('Switch Type')),
                            self.clean_value(row.get('Switch Feel ')),  # Note the space in CSV
                            self.clean_value(row.get('Software Configuration Profiles')),
                            self.clean_value(row.get('Windows')),
                            self.clean_value(row.get('macOS')),
                            self.clean_value(row.get('Linux'))
                        )
                    )
                    
                    self.conn.commit()
                    ingested += 1
                    
                    if ingested % 10 == 0:
                        logger.info(f"Progress: {ingested} keyboards ingested")
                    
                except Exception as e:
                    self.conn.rollback()
                    logger.error(f"Error ingesting keyboard at row {idx}: {e}")
                    failed += 1
                    continue
            
            logger.info(f"Keyboards ingestion complete: {ingested} ingested, {skipped} skipped, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error in keyboards ingestion: {e}")
            raise
    
    def print_summary(self):
        """Print summary statistics of ingested data."""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("DATABASE SUMMARY")
            logger.info("=" * 60)
            
            # Count brands
            self.cursor.execute("SELECT COUNT(*) as count FROM brands")
            brands_count = self.cursor.fetchone()['count']
            logger.info(f"Total Brands: {brands_count}")
            
            # Count products by category
            self.cursor.execute("""
                SELECT category_name, COUNT(*) as count
                FROM products
                GROUP BY category_name
                ORDER BY category_name
            """)
            for row in self.cursor.fetchall():
                logger.info(f"Total {row['category_name']}s: {row['count']}")
            
            # Count total products
            self.cursor.execute("SELECT COUNT(*) as count FROM products")
            products_count = self.cursor.fetchone()['count']
            logger.info(f"Total Products: {products_count}")
            
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
            raise

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING FULL INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432'),
        'sslmode': 'require'  # Required for Google Cloud PostgreSQL
    }
    
    db_instance = ElectronicsDataPipeline(db_config)
    db_instance.connect()
    
    try:
        # Step 1: Ingest brands first (required for foreign keys)
        logger.info("Step 1: Ingesting brands...")
        db_instance.ingest_brands(pd.read_csv("raw_data/brands.csv"))
        
        # Step 2: Ingest monitors
        logger.info("Step 2: Ingesting monitors...")
        db_instance.ingest_monitors(pd.read_csv("raw_data/monitors_clean2.csv"))

        # Step 3: Ingest mice
        logger.info("Step 3: Ingesting mice...")
        db_instance.ingest_mice(pd.read_csv("raw_data/mice_clean2.csv"))
        
        # Step 4: Ingest keyboards
        logger.info("Step 4: Ingesting keyboards...")
        db_instance.ingest_keyboards(pd.read_csv("raw_data/keyboards_clean2.csv"))
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        # Print summary statistics
        db_instance.print_summary()
        
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        db_instance.close()