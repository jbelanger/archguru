from typing import List, Dict, Any, Optional
from .base_client import BaseAPIClient

class StackOverflowClient(BaseAPIClient):
    """Client for StackOverflow API research"""
    
    def __init__(self):
        super().__init__("https://api.stackexchange.com/2.3")
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Unified search interface"""
        return self.search_questions(query, **kwargs)
    
    def search_questions(
        self, 
        query: str, 
        tags: Optional[List[str]] = None,
        sort: str = "votes", 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search StackOverflow questions"""
        
        params = {
            "order": "desc",
            "sort": sort,
            "intitle": query,
            "site": "stackoverflow",
            "pagesize": limit
        }
        
        if tags:
            params["tagged"] = ";".join(tags)
        
        data = self._make_request("search", params)
        
        return [
            {
                "title": q.get("title", ""),
                "score": q.get("score", 0),
                "view_count": q.get("view_count", 0),
                "answer_count": q.get("answer_count", 0),
                "tags": q.get("tags", []),
                "url": q.get("link", ""),
                "is_answered": q.get("is_answered", False)
            }
            for q in data.get("items", [])
        ]