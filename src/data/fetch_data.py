"""
Data Fetching Script for News Headlines

This module scrapes the front page of Hacker News and saves the titles
to a CSV file for later model training.

Usage:
    python -m src.data.fetch_data
"""

import csv
import logging
from logging import config
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import yaml


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

def fetch_html(url: str) -> str:
    """
    Downloads the HTML content from a given URL.
    
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
        return response.text
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise

def parse_titles(html_content: str) -> list:
    """
    Extracts story titles from Hacker News HTML.
    
    Returns:
        list: A list of strings (the headlines).
    """
    soup = BeautifulSoup(html_content, 'html.parser')
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
            f.write(html_content)
    else:
        logger.info(f"Successfully extracted {len(titles)} headlines.")
        
    return titles

def save_to_csv(titles: list, save_path: Path):
    """
    Saves the list of titles to a CSV file.
    
    BEST PRACTICE:
    - Create the parent directory if it doesn't exist.
    - Use 'utf-8' encoding to handle special characters.
    - Write headers (Label, Title) so the training script knows what to expect.
    """
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['label', 'title'])
        
        # For Hacker News, we don't have pre-defined labels. 
        # We will simulate a binary label for educational purposes.
        # Example: If title contains '?' it might be a question (label 1), else 0.
        # This is just to have a target variable for the classifier.
        for title in titles:
            # Simple heuristic: Is it a question?
            label = 1 if '?' in title else 0
            writer.writerow([label, title])
            
    logger.info(f"Data saved successfully to {save_path}")

def main():
    """Main entry point for the script."""
    logger.info("Starting data fetch process...")
    
    # 1. Load Config
    try:
        config = load_config()
    except FileNotFoundError:
        return  # Exit if no config
    
    url = config['data']['url']
    save_path = Path(config['data']['save_path'])
    
    # 2. Fetch HTML
    try:
        html = fetch_html(url)
    except requests.exceptions.RequestException:
        logger.error("Exiting due to fetch failure.")
        return
    
    # 3. Parse Titles
    titles = parse_titles(html)
    
    # 4. Save to CSV
    if titles:
        save_to_csv(titles, save_path)
        logger.info("Data fetch process completed successfully.")
    else:
        logger.error("No data to save. Pipeline stopping.")
        # ⭐ BEST PRACTICE: Exit with a non-zero code so automation (GitHub Actions) knows it failed.
        exit(1)

if __name__ == "__main__":
    main()