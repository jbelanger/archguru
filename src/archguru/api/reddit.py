from typing import List, Dict, Any, Optional
from .base_client import BaseAPIClient

class RedditClient(BaseAPIClient):
    """Client for Reddit community research"""
    
    def __init__(self):
        super().__init__("https://www.reddit.com")
        self.session.headers.update({"User-Agent": "archguru/1.0"})
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Unified search interface"""
        return self.search_discussions(query, **kwargs)
    
    def search_discussions(
        self, 
        query: str, 
        subreddits: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search Reddit discussions"""
        
        if not subreddits:
            subreddits = ["programming", "webdev", "Python", "javascript", "reactjs"]
        
        all_results = []
        
        for subreddit in subreddits[:3]:  # Limit API calls
            subreddit = subreddit.strip().replace("r/", "").replace("/", "")
            if not subreddit:
                continue
                
            data = self._make_request(
                f"r/{subreddit}/search.json",
                params={
                    "q": query,
                    "limit": limit,
                    "sort": "top",
                    "t": "year"
                }
            )
            
            for post in data.get("data", {}).get("children", []):
                post_data = post.get("data", {})
                all_results.append({
                    "title": post_data.get("title", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                    "selftext": post_data.get("selftext", "")[:500]
                })
        
        # Return top results sorted by score
        return sorted(all_results, key=lambda x: x["score"], reverse=True)[:limit]