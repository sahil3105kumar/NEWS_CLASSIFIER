"""
Data Fetching Script for News Headlines

This module scrapes news title from sources mentioned in params.yaml and saves the titles
to a CSV file for later model training.

Usage:
    python -m src.data.fetch_data
"""

import logging
import time
from pathlib import Path
import pandas as pd

import requests
from bs4 import BeautifulSoup
import yaml
import feedparser


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path | None = None) -> dict:
    """
    Loads project configuration from params.yaml.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "params.yaml"
    
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Missing params.yaml at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config

def fetch_and_parse_html(url: str) -> list:
    """
    Downloads the HTML content from a given URL and parses the titles.

    - Use a proper User-Agent header. Some sites block default Python requests.
    - Add error handling for network issues.
    - Add a small delay to be a good internet citizen.
    """

    # Hardcoded User-Agent for now, but could be moved to params.yaml if needed

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        logger.info(f"Fetching data from {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an error for 404 or 500 status codes
        
        # wait 1 second before any further processing
        time.sleep(1) 
        
        logger.info("Successfully fetched HTML content.")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        logger.error(f"Failed to parse HTML from {url}: {e}")
        raise

    titles = []
    
    # Hacker News structure: Titles are inside <span class="titleline"> -> <a>
    # Note: We skip the 'More' link at the bottom.
    title_spans = soup.find_all('span', class_='titleline')
    
    for span in title_spans:
        link = span.find('a')
        if link and link.string:
            # Clean the text: remove extra whitespace, newlines, etc.
            clean_title = ' '.join(link.string.split())
            titles.append(clean_title)
    
    if not titles:
        logger.warning("No titles found. The website structure might have changed.")
        # ⭐ if this happens, we might want to save the raw HTML for debugging and inspect and update accordingly
        with open("debug_raw_html.html", "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        logger.info(f"Successfully extracted {len(titles)} headlines.")
        
    return titles

def fetch_and_parse_rss(url: str) -> list:
    """
    Fetches and parses RSS feed from a given URL.
    
    Returns:
        list: A list of story titles extracted from the RSS feed.
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        logger.info(f"Fetching RSS feed from {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        items = feed.entries
        
        titles = [' '.join(item.title.split()) for item in items if item.title]
        
        logger.info(f"Successfully fetched and parsed RSS feed. Found {len(titles)} titles.")
        return titles
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch RSS feed from {url}: {e}")
        raise


def save_to_csv(titles: list, save_path: Path, mode: str = 'append'):
    """
    Saves the list of titles to a CSV file.
    
    BEST PRACTICES:
    - 'append' mode: Add new data without deleting old data.
    - Deduplication: Avoid inserting the same title multiple times.
    - Timestamping: (Optional) Track when each row was collected.
    - Atomic writes: Write to a temp file first, then rename to avoid corruption.
    
    Args:
        titles: List of headline strings.
        save_path: Path to the CSV file.
        mode: 'overwrite' or 'append' (default: 'append').
    """
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    new_rows = []
    existing_titles = set()
    existing_df = pd.DataFrame()
    combined_df = pd.DataFrame()

    # If appending and file exists, load existing data for deduplication
    if mode == 'append' and save_path.exists():
        try:
            existing_df = pd.read_csv(save_path)
            existing_titles = set(existing_df['title'].tolist())
            logger.info(f"Found {len(existing_titles)} existing headlines in dataset.")
        except Exception as e:
            logger.warning(f"Could not read existing CSV. Will overwrite. Error: {e}")
            mode = 'overwrite'  # Fallback to overwrite if file is corrupt
    
    # Prepare new rows, skipping duplicates
    for title in titles:
        if title not in existing_titles:
            # ⭐ BEST PRACTICE: Add a timestamp column for time-series analysis
            new_rows.append({
                'is_question': 1 if '?' in title else 0, # 1 is for questions, 0 for statements
                'title': title,
                'scraped_at': pd.Timestamp.now().isoformat()
            })
    
    if not new_rows:
        logger.info("No new headlines to add. File unchanged.")
        return
    
    logger.info(f"Adding {len(new_rows)} new headlines to dataset.")
    new_df = pd.DataFrame(new_rows)
    
    if mode == 'append' and save_path.exists():
        # Combine old and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # ⭐ BEST PRACTICE: Write to a temporary file first, then replace

        temp_path = save_path.with_suffix('.csv.tmp')
        combined_df.to_csv(temp_path, index=False, encoding='utf-8')
        temp_path.replace(save_path)
        total_rows = len(combined_df)
        logger.info(f"Dataset updated successfully. Total rows now: {total_rows}")
    else:
        # Overwrite mode (first run)
        new_df.to_csv(save_path, index=False, encoding='utf-8')
        total_rows = len(new_df)
        logger.info(f"Dataset created successfully with {total_rows} rows.")
    

def main():
    """Main entry point for the script."""
    logger.info("Starting data fetch process...")
    
    # 1. Load Config
    try:
        config = load_config()
    except FileNotFoundError:
        return  # Exit if no config
    
    save_path = Path(config['data']['save_path'])

    
    alltitles = []
    # 2. Fetch and parse titles
    
    for source in config['sources']:
        if source['type'] == 'html':
            try:
                titles= fetch_and_parse_html(source['url'])
                alltitles.extend(titles)
            except requests.exceptions.RequestException:
                logger.error(f"Error fetching HTML from {source['url']}")
        else:
            try:
                titles = fetch_and_parse_rss(source['url'])
                alltitles.extend(titles)
            except requests.exceptions.RequestException:
                logger.error(f"Error fetching RSS from {source['url']}")

    
    
    # 3. Save to CSV (append mode by default)
    if alltitles:       
        save_to_csv(alltitles, save_path, mode='append') 
        logger.info("Data fetch process completed successfully.")
    else:
        logger.error("No data to save. Pipeline stopping.")
        exit(1)

if __name__ == "__main__":
    main()