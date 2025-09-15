from typing import List, Dict, Any, Optional
from .base_client import BaseAPIClient

class GitHubClient(BaseAPIClient):
    """Client for GitHub API research"""
    
    def __init__(self):
        super().__init__("https://api.github.com")
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Unified search interface"""
        return self.search_repositories(query, **kwargs)
    
    def search_repositories(
        self, 
        query: str, 
        language: Optional[str] = None,
        sort: str = "stars", 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        
        search_query = f"{query} language:{language}" if language else query
        
        data = self._make_request(
            "search/repositories",
            params={
                "q": search_query,
                "sort": sort,
                "order": "desc",
                "per_page": limit
            }
        )
        
        return [
            {
                "name": repo["full_name"],
                "description": repo.get("description", ""),
                "stars": repo["stargazers_count"],
                "language": repo.get("language"),
                "url": repo["html_url"],
                "topics": repo.get("topics", [])
            }
            for repo in data.get("items", [])
        ]