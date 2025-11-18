import psycopg2
from psycopg2.extras import RealDictCursor
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
        logging.FileHandler('product_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductEnricher:
    """Enrich product data using Perplexity Search + OpenAI extraction"""
    
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
    
    def get_products_to_enrich(self):
        """Get all products that need enrichment (missing price or product_link)"""
        self.cursor.execute("""
            SELECT p.product_id, p.product_name, b.brand_name, p.category_name
            FROM products p
            JOIN brands b ON p.brand_id = b.brand_id
            WHERE p.price IS NULL OR p.product_link IS NULL
            ORDER BY p.product_id
        """)
        products = self.cursor.fetchall()
        logger.info(f"Found {len(products)} products to enrich")
        return products
    
    def search_perplexity(self, product_name: str, brand_name: str, category: str) -> str:
        """
        Use Perplexity Search API to get information about the product
        
        Returns:
            Combined content from search results
        """
        try:
            url = "https://api.perplexity.ai/search"
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            query = f"What is the current price in USD and product page URL for {brand_name} {product_name} {category}?"
            
            payload = {
                "query": query,
                "return_citations": True,
                "return_images": False,
                "recency_filter": "month"  # Recent pricing
            }
            
            logger.debug(f"Calling Perplexity Search API for: {product_name}")
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
                    url_link = r.get('url', '')
                    if snippet:
                        content_pieces.append(f"{snippet}\nURL: {url_link}")
            
            combined_content = '\n\n'.join(content_pieces)
            logger.debug(f"Perplexity content for {product_name}: {combined_content[:200]}...")
            
            return combined_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API request failed for {product_name}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error in Perplexity search for {product_name}: {e}")
            return ""
    
    def extract_with_openai(self, product_name: str, search_content: str) -> dict:
        """
        Use OpenAI to extract structured price and product_link from search results
        
        Args:
            product_name: Name of the product
            search_content: Combined search results from Perplexity
            
        Returns:
            dict with 'price' and 'product_link' keys
        """
        try:
            if not self.openai_model:
                logger.error("OpenAI model not available")
                return {'price': None, 'product_link': None}
            
            if not search_content or len(search_content.strip()) < 20:
                logger.warning(f"Insufficient search content for {product_name}")
                return {'price': None, 'product_link': None}
            
            prompt = f"""Based on the following search results about {product_name}, extract:
1. Current price in USD - just the numeric value (e.g., 299.99)
2. Official product page URL - full URL starting with https://

Search Results:
{search_content}

Respond ONLY in this exact format (no extra text):
Price: [numeric price or "Unknown"]
Link: [URL or "Unknown"]

If you cannot find the information, use "Unknown" for that field.
For price, provide ONLY the number without $ symbol.
Do not include any explanation, just the two lines above."""

            logger.debug(f"Sending to OpenAI for extraction: {product_name}")
            
            # Use invoke() EXACTLY like the example
            response = self.openai_model.invoke(prompt)
            answer = response.content.strip()
            
            logger.debug(f"OpenAI response for {product_name}: {answer}")
            
            # Parse the response
            price = None
            product_link = None
            
            lines = answer.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('price:'):
                    price_str = line.split(':', 1)[1].strip()
                    if price_str.lower() not in ['unknown', 'n/a', 'not found']:
                        # Extract numeric value
                        import re
                        price_match = re.search(r'\d+\.?\d*', price_str)
                        if price_match:
                            try:
                                price = float(price_match.group())
                            except:
                                price = None
                elif line.lower().startswith('link:'):
                    product_link = line.split(':', 1)[1].strip()
                    if product_link.lower() in ['unknown', 'n/a', 'not found']:
                        product_link = None
                    elif product_link and not product_link.startswith('http'):
                        product_link = 'https://' + product_link
            
            logger.info(f"{product_name}: Price=${price}, Link={product_link}")
            return {'price': price, 'product_link': product_link}
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed for {product_name}: {e}")
            return {'price': None, 'product_link': None}
    
    def query_product(self, product_name: str, brand_name: str, category: str) -> dict:
        """
        Main method: Search with Perplexity, then extract with OpenAI
        
        Returns:
            dict with 'price' and 'product_link' keys
        """
        # Step 1: Search with Perplexity
        logger.debug(f"Step 1: Searching Perplexity for {product_name}")
        search_content = self.search_perplexity(product_name, brand_name, category)
        
        if not search_content:
            logger.warning(f"No search results from Perplexity for {product_name}")
            return {'price': None, 'product_link': None}
        
        # Step 2: Extract structured data with OpenAI
        logger.debug(f"Step 2: Extracting data with OpenAI for {product_name}")
        result = self.extract_with_openai(product_name, search_content)
        
        return result
    
    def reconnect_if_needed(self):
        """Reconnect to database if connection is closed"""
        try:
            # Test if connection is alive
            self.cursor.execute("SELECT 1")
        except:
            logger.warning("Database connection lost, reconnecting...")
            self.close()
            self.connect()

    def update_product(self, product_id: int, product_name: str, price: float, product_link: str):
        """Update product with enriched data"""
        try:
            # Reconnect if needed
            self.reconnect_if_needed()
            
            self.cursor.execute(
                """
                UPDATE products 
                SET price = %s, product_link = %s
                WHERE product_id = %s
                """,
                (price, product_link, product_id)
            )
            self.conn.commit()
            logger.info(f"✓ Updated {product_name} (ID: {product_id})")
            
        except Exception as e:
            try:
                self.conn.rollback()
            except:
                pass
            logger.error(f"Failed to update {product_name}: {e}")
            # Try to reconnect for next product
            try:
                self.reconnect_if_needed()
            except:
                pass

    def enrich_all_products(self, delay_seconds: float = 2.0, limit: int = None):
        """
        Enrich all products with missing data
        
        Args:
            delay_seconds: Delay between API calls to respect rate limits
            limit: Optional limit on number of products to process (for testing)
        """
        products = self.get_products_to_enrich()
        
        if not products:
            logger.info("No products need enrichment!")
            return
        
        if limit:
            products = products[:limit]
            logger.info(f"Processing limited to {limit} products")
        
        total = len(products)
        success_count = 0
        partial_count = 0
        failed_count = 0
        skipped_count = 0  # Already enriched
        
        logger.info(f"Starting enrichment for {total} products...")
        logger.info(f"Estimated cost: ${total * 0.004:.2f} - ${total * 0.010:.2f}")
        
        for idx, product in enumerate(products, 1):
            product_id = product['product_id']
            product_name = product['product_name']
            brand_name = product['brand_name']
            category = product['category_name']
            
            # Check if already enriched (both fields filled)
            try:
                self.reconnect_if_needed()
                self.cursor.execute(
                    """
                    SELECT price, product_link 
                    FROM products 
                    WHERE product_id = %s
                    """,
                    (product_id,)
                )
                current = self.cursor.fetchone()
                
                if current and current['price'] is not None and current['product_link'] is not None:
                    logger.info(f"[{idx}/{total}] SKIP: {product_name} (already enriched)")
                    skipped_count += 1
                    continue
            except Exception as e:
                logger.warning(f"Error checking product status: {e}")
            
            logger.info(f"[{idx}/{total}] Processing: {brand_name} {product_name}")
            
            # Query product info (Perplexity + OpenAI)
            result = self.query_product(product_name, brand_name, category)
            
            price = result.get('price')
            product_link = result.get('product_link')
            
            # Update database
            if price or product_link:
                self.update_product(product_id, product_name, price, product_link)
                
                if price and product_link:
                    success_count += 1
                else:
                    partial_count += 1
            else:
                failed_count += 1
                logger.warning(f"No data found for {product_name}")
            
            # Rate limiting delay
            if idx < total:
                time.sleep(delay_seconds)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ENRICHMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total products processed: {total}")
        logger.info(f"Skipped (already enriched): {skipped_count}")
        logger.info(f"Fully enriched (both fields): {success_count}")
        logger.info(f"Partially enriched (one field): {partial_count}")
        logger.info(f"Failed (no data): {failed_count}")
        logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PRODUCT ENRICHMENT PIPELINE")
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
        exit(1)
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in .env file!")
        exit(1)
    
    # Initialize enricher
    enricher = ProductEnricher(db_config, perplexity_api_key, openai_api_key)
    enricher.connect()
    
    try:
        # Run enrichment - you can add limit=10 for testing
        enricher.enrich_all_products(delay_seconds=2.0)
        
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