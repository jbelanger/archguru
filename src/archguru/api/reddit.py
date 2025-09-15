"""
Reddit API client for community research
"""
import requests
from typing import List, Dict, Any
from ..core.config import Config


class RedditClient:
    """Client for Reddit community research"""

    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "archguru/1.0 (research bot)"
        })

    def search_discussions(self, query: str, subreddits: List[str] = None,
                          limit: int = 5) -> List[Dict[str, Any]]:
        """Search Reddit discussions"""
        if subreddits is None:
            subreddits = ["programming", "webdev", "Python", "javascript", "reactjs"]

        results = []

        for subreddit_raw in subreddits[:3]:  # Limit to avoid rate limits
            try:
                # Clean subreddit name - remove any r/ prefix and extra slashes
                subreddit = subreddit_raw.strip().replace("r/", "").replace("/", "")
                if not subreddit:
                    continue

                params = {
                    "q": query,
                    "limit": limit,
                    "sort": "top",
                    "t": "year"
                }

                response = self.session.get(
                    f"{self.base_url}/r/{subreddit}/search.json",
                    params=params,
                    timeout=8
                )
                response.raise_for_status()

                data = response.json()
                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    results.append({
                        "title": post_data.get("title", ""),
                        "subreddit": post_data.get("subreddit", ""),
                        "score": post_data.get("score", 0),
                        "num_comments": post_data.get("num_comments", 0),
                        "url": f"https://reddit.com{post_data.get('permalink', '')}",
                        "selftext": post_data.get("selftext", "")[:500]  # Truncate
                    })

            except Exception as e:
                print(f"Reddit search error for r/{subreddit}: {e}")
                continue

        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

    def get_top_posts(self, subreddit: str, time_period: str = "month",
                      limit: int = 5) -> List[Dict[str, Any]]:
        """Get top posts from a subreddit"""
        try:
            params = {
                "limit": limit,
                "t": time_period
            }

            response = self.session.get(
                f"{self.base_url}/r/{subreddit}/top.json",
                params=params,
                timeout=8
            )
            response.raise_for_status()

            results = []
            data = response.json()
            for post in data.get("data", {}).get("children", []):
                post_data = post.get("data", {})
                results.append({
                    "title": post_data.get("title", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                    "selftext": post_data.get("selftext", "")[:300]
                })

            return results

        except Exception as e:
            print(f"Reddit top posts error: {e}")
            return []