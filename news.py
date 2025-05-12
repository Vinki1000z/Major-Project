"""
News fetching module.
Contains function to fetch related news articles for a stock ticker using Marketaux API with an API key.
"""

import requests

# Function to fetch news articles related to stock symbols
def fetch_news(symbols):
    """
    Fetch latest news articles related to the given stock symbols using Marketaux API.
    symbols: list of stock ticker strings, e.g. ['TSLA', 'AMZN', 'MSFT']
    Returns a list of news items with title and description.
    """
    api_key = "r4QaPN6fLeQR6emY4fM13UircsBnTnB10Ko31GEC"
    symbols_str = ",".join(symbols)
    # Construct API URL with query parameters
    url = f"https://api.marketaux.com/v1/news/all?symbols={symbols_str}&filter_entities=true&language=en&api_token={api_key}&limit=10"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            news_items = []
            # Parse news articles from response
            for article in data.get('data', []):
                news_items.append({
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', '')
                })
            return news_items
        else:
            print(f"Marketaux API request failed with status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Marketaux news: {e}")
        return []
