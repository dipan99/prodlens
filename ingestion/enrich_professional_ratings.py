import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import re

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('professional_ratings_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProfessionalRatingsEnricher:
    """Enrich professional ratings with TRUE batching"""
    
    def __init__(self, db_config: dict, perplexity_api_key: str, openai_api_key: str):
        self.db_config = db_config
        self.perplexity_api_key = perplexity_api_key
        self.conn = None
        self.cursor = None
        
        try:
            logger.info("Initializing OpenAI model...")
            os.environ['OPENAI_API_KEY'] = openai_api_key
            self.openai_model = init_chat_model("gpt-4o-mini", model_provider="openai")
            logger.info("OpenAI model initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI initialization error: {e}")
            self.openai_model = None
        
    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def reconnect_if_needed(self):
        try:
            self.cursor.execute("SELECT 1")
        except:
            logger.warning("Database connection lost, reconnecting...")
            self.close()
            self.connect()
    
    def get_products_to_enrich(self):
        """Get products WITHOUT professional ratings"""
        self.cursor.execute("""
            SELECT p.product_id, p.product_name, b.brand_name, p.category_name,
                   p.ranking_general, p.ranking_gaming, p.ranking_office, p.ranking_editing
            FROM products p
            JOIN brands b ON p.brand_id = b.brand_id
            LEFT JOIN professional_ratings pr ON p.product_id = pr.product_id
            WHERE pr.rating_id IS NULL
              AND (p.ranking_general IS NOT NULL 
                   OR p.ranking_gaming IS NOT NULL 
                   OR p.ranking_office IS NOT NULL 
                   OR p.ranking_editing IS NOT NULL)
            ORDER BY p.product_id
        """)
        products = self.cursor.fetchall()
        logger.info(f"Found {len(products)} products WITHOUT professional ratings")
        return products
    
    def search_perplexity_batch(self, products_batch: list) -> str:
        """
        Single Perplexity search for entire batch
        
        Returns:
            Combined search content for all products
        """
        try:
            url = "https://api.perplexity.ai/search"
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            # Create query for all products in batch
            product_list = []
            for i, p in enumerate(products_batch, 1):
                product_list.append(f"{i}. {p['brand_name']} {p['product_name']} {p['category_name']}")
            
            query = f"Find RTINGS reviews with pros, cons, and summary for these products:\n" + "\n".join(product_list)
            
            payload = {
                "query": query,
                "return_citations": True,
                "return_images": False,
                "recency_filter": "year"
            }
            
            logger.info(f"Batch Perplexity search for {len(products_batch)} products...")
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract all content
            full_content = ""
            
            if 'answer' in result and result['answer']:
                full_content += result['answer'] + "\n\n"
            
            if 'results' in result and result['results']:
                for r in result['results']:
                    snippet = r.get('snippet', '') or r.get('content', '') or r.get('description', '')
                    if snippet:
                        full_content += snippet + "\n\n"
            
            logger.debug(f"Perplexity returned {len(full_content)} characters for batch")
            
            return full_content
            
        except Exception as e:
            logger.error(f"Perplexity batch search failed: {e}")
            return ""
    
    def extract_batch_with_openai(self, products_batch: list, search_content: str) -> dict:
        """
        Single OpenAI call to extract data for ALL products in batch
        
        Returns:
            dict mapping product_id to extracted data
        """
        try:
            if not self.openai_model or not search_content:
                logger.error("OpenAI model unavailable or no content")
                return {}
            
            # Build prompt with all products
            product_descriptions = []
            for i, p in enumerate(products_batch, 1):
                product_descriptions.append(f"{i}. {p['brand_name']} {p['product_name']} (Product ID: {p['product_id']})")
            
            prompt = f"""From the search results below, extract RTINGS review information for EACH of these products:

Products to extract:
{chr(10).join(product_descriptions)}

Search Results:
{search_content}

For EACH product listed above, extract:
- PROS: Positive points (bullet format, max 5)
- CONS: Negative points (bullet format, max 5)
- SUMMARY: 2-3 sentence summary
- REVIEW_URL: RTINGS URL if found
- REVIEW_DATE: Date in YYYY-MM-DD if found

Format your response EXACTLY like this:

PRODUCT 1 (ID: [product_id]):
PROS:
- [point 1]
- [point 2]

CONS:
- [point 1]
- [point 2]

SUMMARY:
[summary text]

REVIEW_URL:
[URL or "Not found"]

REVIEW_DATE:
[YYYY-MM-DD or "Not found"]

---

PRODUCT 2 (ID: [product_id]):
[same format]

---

If you cannot find information for a specific product, still include its section with "Not found" for each field.
IMPORTANT: Extract information for ALL {len(products_batch)} products listed above."""

            logger.info(f"Single OpenAI call for {len(products_batch)} products...")
            
            response = self.openai_model.invoke(prompt)
            answer = response.content.strip()
            
            logger.debug(f"OpenAI batch response: {answer[:500]}...")
            
            # Parse response for each product
            results = {}
            
            # Split by product sections
            product_sections = re.split(r'PRODUCT \d+ \(ID: (\d+)\):', answer)
            
            # product_sections will be: ['', product_id1, content1, product_id2, content2, ...]
            for i in range(1, len(product_sections), 2):
                if i + 1 >= len(product_sections):
                    break
                
                product_id = int(product_sections[i])
                section_content = product_sections[i + 1]
                
                # Parse this product's data
                pros = None
                cons = None
                summary = None
                review_url = None
                review_date = None
                
                parts = section_content.split('---')[0]  # Remove next product separator
                
                # Extract PROS
                pros_match = re.search(r'PROS:\s*\n((?:- .+\n?)+)', parts, re.IGNORECASE)
                if pros_match:
                    pros_lines = [line.strip('- ').strip() for line in pros_match.group(1).split('\n') if line.strip().startswith('-')]
                    pros = '\n'.join(pros_lines) if pros_lines else None
                
                # Extract CONS
                cons_match = re.search(r'CONS:\s*\n((?:- .+\n?)+)', parts, re.IGNORECASE)
                if cons_match:
                    cons_lines = [line.strip('- ').strip() for line in cons_match.group(1).split('\n') if line.strip().startswith('-')]
                    cons = '\n'.join(cons_lines) if cons_lines else None
                
                # Extract SUMMARY
                summary_match = re.search(r'SUMMARY:\s*\n(.+?)(?=\n\w+:|$)', parts, re.IGNORECASE | re.DOTALL)
                if summary_match:
                    summary_text = summary_match.group(1).strip()
                    if summary_text.lower() != 'not found':
                        summary = summary_text
                
                # Extract REVIEW_URL
                url_match = re.search(r'REVIEW_URL:\s*\n(.+)', parts, re.IGNORECASE)
                if url_match:
                    url_text = url_match.group(1).strip()
                    if url_text.lower() not in ['not found', 'n/a']:
                        review_url = url_text
                
                # Extract REVIEW_DATE
                date_match = re.search(r'REVIEW_DATE:\s*\n(.+)', parts, re.IGNORECASE)
                if date_match:
                    date_text = date_match.group(1).strip()
                    if date_text.lower() not in ['not found', 'n/a']:
                        try:
                            review_date = datetime.strptime(date_text, '%Y-%m-%d').date()
                        except:
                            pass
                
                results[product_id] = {
                    'pros': pros,
                    'cons': cons,
                    'summary': summary,
                    'review_url': review_url,
                    'review_date': review_date
                }
            
            logger.info(f"Extracted data for {len(results)}/{len(products_batch)} products")
            
            return results
            
        except Exception as e:
            logger.error(f"OpenAI batch extraction failed: {e}")
            return {}
    
    def insert_professional_rating(self, product_id: int, product_name: str, 
                                   ranking_general: float, ranking_gaming: float,
                                   ranking_office: float, ranking_editing: float,
                                   review_details: dict):
        """Insert professional rating"""
        try:
            self.reconnect_if_needed()
            
            self.cursor.execute(
                """
                INSERT INTO professional_ratings (
                    product_id, reviewer_website,
                    rating_general, rating_gaming, rating_office, rating_editing,
                    pros, cons, summary, review_url, review_date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    product_id,
                    'https://www.rtings.com',
                    ranking_general,
                    ranking_gaming,
                    ranking_office,
                    ranking_editing,
                    review_details.get('pros'),
                    review_details.get('cons'),
                    review_details.get('summary'),
                    review_details.get('review_url'),
                    review_details.get('review_date')
                )
            )
            self.conn.commit()
            logger.info(f"✓ Inserted rating for {product_name}")
            return True
            
        except Exception as e:
            try:
                self.conn.rollback()
            except:
                pass
            logger.error(f"Failed to insert {product_name}: {e}")
            return False
    
    def enrich_all_ratings(self, delay_seconds: float = 5.0, batch_size: int = 5, limit: int = None):
        """
        TRUE BATCHING: 1 Perplexity + 1 OpenAI call per batch
        
        Args:
            delay_seconds: Delay between batches
            batch_size: Products per batch (5-10 recommended)
            limit: Optional limit on total products
        """
        products = self.get_products_to_enrich()
        
        if not products:
            logger.info("No products need professional ratings!")
            return
        
        if limit:
            products = products[:limit]
        
        total = len(products)
        success_count = 0
        failed_count = 0
        
        num_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Starting TRUE BATCH enrichment for {total} products")
        logger.info(f"Batch size: {batch_size} products")
        logger.info(f"Total batches: {num_batches}")
        logger.info(f"API calls: {num_batches} Perplexity + {num_batches} OpenAI")
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = products[batch_start:batch_end]
            
            batch_num = batch_start // batch_size + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_num}/{num_batches}: Products {batch_start+1}-{batch_end}")
            logger.info(f"{'='*60}")
            
            # Step 1: Single Perplexity search for entire batch
            search_content = self.search_perplexity_batch(batch)
            
            if not search_content:
                logger.warning(f"Batch {batch_num}: No search content, skipping")
                failed_count += len(batch)
                continue
            
            # Step 2: Single OpenAI call to extract ALL products
            batch_results = self.extract_batch_with_openai(batch, search_content)
            
            if not batch_results:
                logger.warning(f"Batch {batch_num}: No extraction results, skipping")
                failed_count += len(batch)
                continue
            
            # Step 3: Insert each product's data
            for product in batch:
                product_id = product['product_id']
                product_name = product['product_name']
                
                review_details = batch_results.get(product_id)
                
                if not review_details:
                    logger.warning(f"No data extracted for {product_name}")
                    failed_count += 1
                    continue
                
                success = self.insert_professional_rating(
                    product_id, product_name,
                    product['ranking_general'], product['ranking_gaming'],
                    product['ranking_office'], product['ranking_editing'],
                    review_details
                )
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            
            # Delay between batches
            if batch_end < total:
                logger.info(f"Waiting {delay_seconds}s before next batch...")
                time.sleep(delay_seconds)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PROFESSIONAL RATINGS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total products: {total}")
        logger.info(f"Successfully enriched: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Perplexity calls made: {num_batches}")
        logger.info(f"OpenAI calls made: {num_batches}")
        logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PROFESSIONAL RATINGS - TRUE BATCH MODE")
    logger.info("=" * 60)
    
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432'),
        'sslmode': 'require'
    }
    
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not perplexity_api_key or not openai_api_key:
        logger.error("API keys not found!")
        exit(1)
    
    enricher = ProfessionalRatingsEnricher(db_config, perplexity_api_key, openai_api_key)
    enricher.connect()
    
    try:
        # TRUE batching: 5 products = 1 Perplexity + 1 OpenAI call
        enricher.enrich_all_ratings(delay_seconds=5.0, batch_size=5)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"Failed: {e}")
    finally:
        enricher.close()