"""
News fetching module.
Contains function to fetch related news articles for a stock ticker using a free public API.
"""

import requests

def fetch_news(ticker):
    """
    Fetch latest news articles related to the stock ticker using NewsAPI.org free demo endpoint.
    Returns a list of news items with title, publisher, link, and published date.
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=5&apiKey=demo"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            news_items = []
            for article in articles:
                news_items.append({
                    'title': article.get('title', 'No Title'),
                    'publisher': article.get('source', {}).get('name', ''),
                    'link': article.get('url', ''),
                    'publishedAt': article.get('publishedAt', '')
                })
            return news_items
        else:
            print(f"News API request failed with status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
