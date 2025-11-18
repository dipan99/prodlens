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
        logging.FileHandler('reviews_enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ReviewsEnricher:
    """Enrich reviews with TRUE batching"""
    
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
            logger.error(f"Failed to connect: {e}")
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
            logger.warning("Reconnecting...")
            self.close()
            self.connect()
    
    def get_products_to_enrich(self, min_reviews: int = 5):
        """Get products with < min_reviews"""
        self.cursor.execute(f"""
            SELECT p.product_id, p.product_name, b.brand_name, p.category_name,
                   COUNT(r.review_id) as review_count
            FROM products p
            JOIN brands b ON p.brand_id = b.brand_id
            LEFT JOIN reviews r ON p.product_id = r.product_id
            GROUP BY p.product_id, p.product_name, b.brand_name, p.category_name
            HAVING COUNT(r.review_id) < {min_reviews}
            ORDER BY p.product_id
        """)
        products = self.cursor.fetchall()
        logger.info(f"Found {len(products)} products with < {min_reviews} reviews")
        return products
    
    def search_perplexity_batch(self, products_batch: list) -> str:
        """Single Perplexity search for entire batch"""
        try:
            url = "https://api.perplexity.ai/search"
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            product_list = []
            for i, p in enumerate(products_batch, 1):
                product_list.append(f"{i}. {p['brand_name']} {p['product_name']} {p['category_name']}")
            
            query = f"Find customer reviews for these products:\n" + "\n".join(product_list)
            
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
            
            # Extract content
            full_content = ""
            
            if 'answer' in result and result['answer']:
                full_content += result['answer'] + "\n\n"
            
            if 'results' in result and result['results']:
                for r in result['results']:
                    snippet = r.get('snippet', '') or r.get('content', '')
                    url_link = r.get('url', '')
                    
                    source = 'Unknown'
                    if 'amazon.com' in url_link.lower():
                        source = 'Amazon'
                    elif 'bestbuy.com' in url_link.lower():
                        source = 'Best Buy'
                    elif 'reddit.com' in url_link.lower():
                        source = 'Reddit'
                    
                    if snippet:
                        full_content += f"[Source: {source}]\n{snippet}\n\n"
            
            return full_content
            
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return ""
    
    def extract_batch_with_openai(self, products_batch: list, search_content: str) -> dict:
        """
        Single OpenAI call to extract reviews for ALL products
        
        Returns:
            dict mapping product_id to list of reviews
        """
        try:
            if not self.openai_model or not search_content:
                return {}
            
            # Build prompt with all products
            product_descriptions = []
            for i, p in enumerate(products_batch, 1):
                needed = 5 - p['review_count']
                product_descriptions.append(
                    f"{i}. {p['brand_name']} {p['product_name']} (Product ID: {p['product_id']}, needs {needed} reviews)"
                )
            
            prompt = f"""From the search results below, extract customer reviews for EACH of these products:

Products to extract:
{chr(10).join(product_descriptions)}

Search Results:
{search_content}

For EACH product, extract up to 5 reviews. Format EXACTLY like this:

PRODUCT 1 (ID: [product_id]):
REVIEW 1:
Rating: [1-5]
Title: [title]
Text: [minimum 20 words]
Source: [Amazon/Best Buy/Reddit/etc]
Date: [YYYY-MM-DD or Unknown]
Verified: [Yes/No/Unknown]
Helpful: [number or 0]

REVIEW 2:
[same format]

---

PRODUCT 2 (ID: [product_id]):
[same format]

---

IMPORTANT: 
- Only extract REAL reviews from the search results
- Each review text must be at least 20 words
- Match reviews to the CORRECT product
- If you can't find reviews for a product, write "No reviews found"

Extract reviews for ALL {len(products_batch)} products."""

            logger.info(f"Single OpenAI call for {len(products_batch)} products...")
            
            response = self.openai_model.invoke(prompt)
            answer = response.content.strip()
            
            # Parse response
            results = {}
            
            # Split by product sections
            product_sections = re.split(r'PRODUCT \d+ \(ID: (\d+)\):', answer)
            
            for i in range(1, len(product_sections), 2):
                if i + 1 >= len(product_sections):
                    break
                
                product_id = int(product_sections[i])
                section_content = product_sections[i + 1]
                
                # Extract reviews for this product
                reviews = []
                
                # Split by reviews
                review_blocks = re.split(r'REVIEW \d+:', section_content)
                
                for block in review_blocks[1:]:
                    if not block.strip() or 'No reviews found' in block:
                        continue
                    
                    review = {
                        'rating': None,
                        'title': None,
                        'text': None,
                        'source': 'Unknown',
                        'date': None,
                        'verified': False,
                        'helpful': 0
                    }
                    
                    lines = block.split('---')[0].strip().split('\n')
                    i = 0
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        if line.lower().startswith('rating:'):
                            try:
                                rating = int(re.search(r'\d', line.split(':', 1)[1]).group())
                                if 1 <= rating <= 5:
                                    review['rating'] = rating
                            except:
                                pass
                        
                        elif line.lower().startswith('title:'):
                            title = line.split(':', 1)[1].strip()
                            if title.lower() not in ['unknown', 'n/a']:
                                review['title'] = title[:255]
                        
                        elif line.lower().startswith('text:'):
                            text = line.split(':', 1)[1].strip()
                            # Read continuation lines
                            j = i + 1
                            while j < len(lines):
                                next_line = lines[j].strip()
                                if ':' in next_line and next_line.split(':')[0].lower() in ['source', 'date', 'verified', 'helpful', 'review']:
                                    break
                                text += ' ' + next_line
                                j += 1
                            
                            if len(text.split()) >= 20:
                                review['text'] = text
                        
                        elif line.lower().startswith('source:'):
                            source = line.split(':', 1)[1].strip()
                            if source.lower() not in ['unknown', 'n/a']:
                                review['source'] = source[:100]
                        
                        elif line.lower().startswith('date:'):
                            date_str = line.split(':', 1)[1].strip()
                            if date_str.lower() not in ['unknown', 'n/a']:
                                try:
                                    review['date'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                                except:
                                    pass
                        
                        elif line.lower().startswith('verified:'):
                            verified_str = line.split(':', 1)[1].strip().lower()
                            review['verified'] = verified_str in ['yes', 'true', 'verified']
                        
                        elif line.lower().startswith('helpful:'):
                            try:
                                review['helpful'] = int(re.search(r'\d+', line.split(':', 1)[1]).group())
                            except:
                                review['helpful'] = 0
                        
                        i += 1
                    
                    if review['rating'] and review['text']:
                        reviews.append(review)
                
                results[product_id] = reviews
            
            total_reviews = sum(len(r) for r in results.values())
            logger.info(f"Extracted {total_reviews} reviews across {len(results)} products")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return {}
    
    def insert_reviews(self, product_id: int, product_name: str, reviews: list) -> int:
        """Insert reviews"""
        inserted = 0
        
        for review in reviews:
            try:
                self.reconnect_if_needed()
                
                review_date = review.get('date') or datetime.now().date()
                
                self.cursor.execute(
                    """
                    INSERT INTO reviews (
                        product_id, user_id, rating, review_title, review_text,
                        source, verified_purchase, helpful_count, review_date
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        product_id,
                        None,
                        review['rating'],
                        review.get('title'),
                        review['text'],
                        review.get('source', 'Unknown'),
                        review.get('verified', False),
                        review.get('helpful', 0),
                        review_date
                    )
                )
                self.conn.commit()
                inserted += 1
                
            except Exception as e:
                try:
                    self.conn.rollback()
                except:
                    pass
                logger.error(f"Insert failed: {e}")
        
        if inserted > 0:
            logger.info(f"✓ Inserted {inserted} reviews for {product_name}")
        
        return inserted
    
    def enrich_all_reviews(self, delay_seconds: float = 5.0, batch_size: int = 5, 
                          limit: int = None, target_reviews: int = 5):
        """TRUE BATCHING: 1 Perplexity + 1 OpenAI per batch"""
        products = self.get_products_to_enrich(min_reviews=target_reviews)
        
        if not products:
            logger.info(f"All products have {target_reviews}+ reviews!")
            return
        
        if limit:
            products = products[:limit]
        
        total = len(products)
        success = 0
        partial = 0
        failed = 0
        total_inserted = 0
        
        num_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Starting TRUE BATCH enrichment for {total} products")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Total batches: {num_batches}")
        logger.info(f"API calls: {num_batches} Perplexity + {num_batches} OpenAI")
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = products[batch_start:batch_end]
            
            batch_num = batch_start // batch_size + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_num}/{num_batches}: Products {batch_start+1}-{batch_end}")
            logger.info(f"{'='*60}")
            
            # Step 1: Single Perplexity search
            search_content = self.search_perplexity_batch(batch)
            
            if not search_content:
                failed += len(batch)
                continue
            
            # Step 2: Single OpenAI extraction for ALL products
            batch_results = self.extract_batch_with_openai(batch, search_content)
            
            if not batch_results:
                failed += len(batch)
                continue
            
            # Step 3: Insert each product's reviews
            for product in batch:
                product_id = product['product_id']
                product_name = product['product_name']
                needed = target_reviews - product['review_count']
                
                reviews = batch_results.get(product_id, [])
                
                if not reviews:
                    failed += 1
                    continue
                
                inserted = self.insert_reviews(product_id, product_name, reviews)
                total_inserted += inserted
                
                if inserted >= needed:
                    success += 1
                elif inserted > 0:
                    partial += 1
                else:
                    failed += 1
            
            if batch_end < total:
                logger.info(f"Waiting {delay_seconds}s...")
                time.sleep(delay_seconds)
        
        logger.info("\n" + "=" * 60)
        logger.info("REVIEWS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total products: {total}")
        logger.info(f"Fully enriched: {success}")
        logger.info(f"Partially enriched: {partial}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total reviews inserted: {total_inserted}")
        logger.info(f"Perplexity calls made: {num_batches}")
        logger.info(f"OpenAI calls made: {num_batches}")
        logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("REVIEWS - TRUE BATCH MODE")
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
    
    enricher = ReviewsEnricher(db_config, perplexity_api_key, openai_api_key)
    enricher.connect()
    
    try:
        # TRUE batching: 5 products = 1 Perplexity + 1 OpenAI
        enricher.enrich_all_reviews(delay_seconds=5.0, batch_size=5, target_reviews=5)
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"Failed: {e}")
    finally:
        enricher.close()