"""
Brand Data Enrichment Script
Uses Perplexity API to fetch country of origin and website URL for each brand
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brand_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BrandEnricher:
    """Enrich brand data using Perplexity API"""
    
    def __init__(self, db_config: dict, perplexity_api_key: str):
        self.db_config = db_config
        self.perplexity_api_key = perplexity_api_key
        self.conn = None
        self.cursor = None
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def get_brands_to_enrich(self):
        """Get all brands that need enrichment (missing country or website)"""
        self.cursor.execute("""
            SELECT brand_id, brand_name 
            FROM brands 
            WHERE country_origin IS NULL OR website_url IS NULL
            ORDER BY brand_name
        """)
        brands = self.cursor.fetchall()
        logger.info(f"Found {len(brands)} brands to enrich")
        return brands
    
    def query_perplexity(self, brand_name: str) -> dict:
        """
        Query Perplexity API for brand information
        
        Returns:
            dict with 'country' and 'website' keys
        """
        try:
            # Craft a specific query for factual information
            query = f"""What is the country of origin (headquarters location) and official website URL for the company "{brand_name}" that manufactures computer peripherals and electronics? 

Please provide:
1. Country of origin (just the country name)
2. Official website URL (full URL starting with https://)

Format your response as:
Country: [country name]
Website: [url]"""

            payload = {
                "model": "llama-3.1-sonar-large-128k-online",
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.1,  # Low temperature for factual responses
                "max_tokens": 200
            }
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.perplexity_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            response_text = result['choices'][0]['message']['content']
            logger.debug(f"Perplexity response for {brand_name}: {response_text}")
            
            # Parse the response
            parsed = self.parse_response(response_text, brand_name)
            return parsed
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {brand_name}: {e}")
            return {'country': None, 'website': None}
        except Exception as e:
            logger.error(f"Error querying Perplexity for {brand_name}: {e}")
            return {'country': None, 'website': None}
    
    def parse_response(self, response_text: str, brand_name: str) -> dict:
        """
        Parse Perplexity response to extract country and website
        
        Args:
            response_text: Raw response from Perplexity
            brand_name: Brand name for logging
            
        Returns:
            dict with 'country' and 'website' keys
        """
        country = None
        website = None
        
        try:
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for country
                if line.lower().startswith('country:'):
                    country = line.split(':', 1)[1].strip()
                    # Remove any extra punctuation
                    country = country.rstrip('.,;')
                
                # Look for website
                elif line.lower().startswith('website:'):
                    website = line.split(':', 1)[1].strip()
                    # Clean up website URL
                    website = website.rstrip('.,;')
                    # Ensure it starts with http
                    if website and not website.startswith('http'):
                        website = 'https://' + website
            
            # Fallback: try to extract from unstructured text
            if not country or not website:
                # Simple extraction if format is different
                lower_text = response_text.lower()
                
                # Try to find website URLs
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                urls = re.findall(url_pattern, response_text)
                if urls and not website:
                    website = urls[0]
                
                # Country extraction is harder, log for manual review if needed
                if not country:
                    logger.warning(f"Could not parse country for {brand_name}. Response: {response_text}")
            
            logger.info(f"{brand_name}: Country={country}, Website={website}")
            
        except Exception as e:
            logger.error(f"Error parsing response for {brand_name}: {e}")
        
        return {
            'country': country,
            'website': website
        }
    
    def update_brand(self, brand_id: int, brand_name: str, country: str, website: str):
        """Update brand with enriched data"""
        try:
            self.cursor.execute(
                """
                UPDATE brands 
                SET country_origin = %s, website_url = %s
                WHERE brand_id = %s
                """,
                (country, website, brand_id)
            )
            self.conn.commit()
            logger.info(f"Updated {brand_name} (ID: {brand_id})")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to update {brand_name}: {e}")
    
    def enrich_all_brands(self, delay_seconds: float = 1.0):
        """
        Enrich all brands with missing data
        
        Args:
            delay_seconds: Delay between API calls to respect rate limits
        """
        brands = self.get_brands_to_enrich()
        
        if not brands:
            logger.info("No brands need enrichment!")
            return
        
        total = len(brands)
        success_count = 0
        partial_count = 0
        failed_count = 0
        
        logger.info(f"Starting enrichment for {total} brands...")
        logger.info(f"Estimated cost: ${total * 0.001:.2f} - ${total * 0.002:.2f}")
        
        for idx, brand in enumerate(brands, 1):
            brand_id = brand['brand_id']
            brand_name = brand['brand_name']
            
            logger.info(f"[{idx}/{total}] Processing: {brand_name}")
            
            # Query Perplexity
            result = self.query_perplexity(brand_name)
            
            country = result.get('country')
            website = result.get('website')
            
            # Update database
            if country or website:
                self.update_brand(brand_id, brand_name, country, website)
                
                if country and website:
                    success_count += 1
                else:
                    partial_count += 1
            else:
                failed_count += 1
                logger.warning(f"No data found for {brand_name}")
            
            # Rate limiting delay
            if idx < total:  # Don't delay after last brand
                time.sleep(delay_seconds)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ENRICHMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total brands processed: {total}")
        logger.info(f"Fully enriched (both fields): {success_count}")
        logger.info(f"Partially enriched (one field): {partial_count}")
        logger.info(f"Failed (no data): {failed_count}")
        logger.info("=" * 60)
    
    def show_sample(self, limit: int = 10):
        """Show sample of enriched brands"""
        self.cursor.execute("""
            SELECT brand_name, country_origin, website_url
            FROM brands
            WHERE country_origin IS NOT NULL OR website_url IS NOT NULL
            ORDER BY brand_name
            LIMIT %s
        """, (limit,))
        
        brands = self.cursor.fetchall()
        
        if brands:
            logger.info(f"\nSample of enriched brands (showing {len(brands)}):")
            logger.info("-" * 80)
            for brand in brands:
                logger.info(f"{brand['brand_name']:20} | {brand['country_origin'] or 'N/A':15} | {brand['website_url'] or 'N/A'}")
            logger.info("-" * 80)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BRAND ENRICHMENT PIPELINE")
    logger.info("=" * 60)
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432'),
        'sslmode': 'require'
    }
    
    # Get Perplexity API key
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
    
    if not perplexity_api_key:
        logger.error("PERPLEXITY_API_KEY not found in .env file!")
        logger.error("Please add: PERPLEXITY_API_KEY=your-api-key-here")
        exit(1)
    
    # Initialize enricher
    enricher = BrandEnricher(db_config, perplexity_api_key)
    enricher.connect()
    
    try:
        # Run enrichment
        enricher.enrich_all_brands(delay_seconds=1.0)  # 1 second delay between calls
        
        # Show sample results
        enricher.show_sample(limit=10)
        
    except KeyboardInterrupt:
        logger.info("\nEnrichment interrupted by user")
    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        raise
    finally:
        enricher.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("=" * 60)