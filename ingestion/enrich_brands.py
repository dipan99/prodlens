import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import requests
import os
import time
import logging
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

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
    """Enrich brand data using Perplexity Search + OpenAI extraction"""
    
    def __init__(self, db_config: dict, perplexity_api_key: str, openai_api_key: str):
        self.db_config = db_config
        self.perplexity_api_key = perplexity_api_key
        self.conn = None
        self.cursor = None
        
        # Initialize OpenAI via LangChain - EXACTLY like the example
        try:
            logger.info("Initializing OpenAI model...")
            os.environ['OPENAI_API_KEY'] = openai_api_key
            self.openai_model = init_chat_model("gpt-4o-mini", model_provider="openai")
            logger.info("OpenAI model initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI initialization error: {e}")
            self.openai_model = None
        
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
    
    def search_perplexity(self, brand_name: str) -> str:
        """
        Use Perplexity Search API to get information about the brand
        
        Returns:
            Combined content from search results
        """
        try:
            url = "https://api.perplexity.ai/search"
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            query = f"What is the country of origin (headquarters) and official website URL for {brand_name} electronics company?"
            
            payload = {
                "query": query,
                "return_citations": True,
                "return_images": False,
                "recency_filter": "year"
            }
            
            logger.debug(f"Calling Perplexity Search API for: {brand_name}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract content from search results
            content_pieces = []
            
            # Get direct answer if available
            if 'answer' in result and result['answer']:
                content_pieces.append(result['answer'])
            
            # Get snippets from top results
            if 'results' in result and result['results']:
                for r in result['results'][:3]:  # Top 3 results
                    snippet = r.get('snippet', '') or r.get('content', '') or r.get('description', '')
                    if snippet:
                        content_pieces.append(snippet)
            
            combined_content = '\n\n'.join(content_pieces)
            logger.debug(f"Perplexity content for {brand_name}: {combined_content[:200]}...")
            
            return combined_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API request failed for {brand_name}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error in Perplexity search for {brand_name}: {e}")
            return ""
    
    def extract_with_openai(self, brand_name: str, search_content: str) -> dict:
        """
        Use OpenAI to extract structured country and website from search results
        
        Args:
            brand_name: Name of the brand
            search_content: Combined search results from Perplexity
            
        Returns:
            dict with 'country' and 'website' keys
        """
        try:
            if not self.openai_model:
                logger.error("OpenAI model not available")
                return {'country': None, 'website': None}
            
            if not search_content or len(search_content.strip()) < 20:
                logger.warning(f"Insufficient search content for {brand_name}")
                return {'country': None, 'website': None}
            
            prompt = f"""Based on the following search results about {brand_name}, extract:
1. Country of origin (headquarters location) - just the country name
2. Official website URL - full URL starting with https://

Search Results:
{search_content}

Respond ONLY in this exact format (no extra text):
Country: [country name or "Unknown"]
Website: [URL or "Unknown"]

If you cannot find the information, use "Unknown" for that field.
Do not include any explanation, just the two lines above."""

            logger.debug(f"Sending to OpenAI for extraction: {brand_name}")
            
            # Use invoke() EXACTLY like the example
            response = self.openai_model.invoke(prompt)
            answer = response.content.strip()
            
            logger.debug(f"OpenAI response for {brand_name}: {answer}")
            
            # Parse the response
            country = None
            website = None
            
            lines = answer.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('country:'):
                    country = line.split(':', 1)[1].strip()
                    if country.lower() in ['unknown', 'n/a', 'not found']:
                        country = None
                elif line.lower().startswith('website:'):
                    website = line.split(':', 1)[1].strip()
                    if website.lower() in ['unknown', 'n/a', 'not found']:
                        website = None
                    elif website and not website.startswith('http'):
                        website = 'https://' + website
            
            logger.info(f"{brand_name}: Country={country}, Website={website}")
            return {'country': country, 'website': website}
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed for {brand_name}: {e}")
            return {'country': None, 'website': None}
    
    def query_brand(self, brand_name: str) -> dict:
        """
        Main method: Search with Perplexity, then extract with OpenAI
        
        Returns:
            dict with 'country' and 'website' keys
        """
        # Step 1: Search with Perplexity
        logger.debug(f"Step 1: Searching Perplexity for {brand_name}")
        search_content = self.search_perplexity(brand_name)
        
        if not search_content:
            logger.warning(f"No search results from Perplexity for {brand_name}")
            return {'country': None, 'website': None}
        
        # Step 2: Extract structured data with OpenAI
        logger.debug(f"Step 2: Extracting data with OpenAI for {brand_name}")
        result = self.extract_with_openai(brand_name, search_content)
        
        return result
    
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
            logger.info(f"✓ Updated {brand_name} (ID: {brand_id})")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to update {brand_name}: {e}")
    
    def enrich_all_brands(self, delay_seconds: float = 1.5):
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
        logger.info(f"Estimated cost: ${total * 0.002:.2f} - ${total * 0.005:.2f}")
        
        for idx, brand in enumerate(brands, 1):
            brand_id = brand['brand_id']
            brand_name = brand['brand_name']
            
            logger.info(f"[{idx}/{total}] Processing: {brand_name}")
            
            # Query brand info (Perplexity + OpenAI)
            result = self.query_brand(brand_name)
            
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
            if idx < total:
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


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BRAND ENRICHMENT PIPELINE")
    logger.info("Perplexity Search + OpenAI Extraction")
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
    
    # Get API keys
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not perplexity_api_key:
        logger.error("PERPLEXITY_API_KEY not found in .env file!")
        logger.error("Please add: PERPLEXITY_API_KEY=your-api-key-here")
        exit(1)
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in .env file!")
        logger.error("Please add: OPENAI_API_KEY=your-api-key-here")
        exit(1)
    
    # Initialize enricher
    enricher = BrandEnricher(db_config, perplexity_api_key, openai_api_key)
    enricher.connect()
    
    try:
        # Run enrichment
        enricher.enrich_all_brands(delay_seconds=1.5)
        
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