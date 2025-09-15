"""
GitHub API client for repository research
"""
import requests
from typing import List, Dict, Any, Optional
from ..core.config import Config


class GitHubClient:
    """Client for GitHub API research"""

    def __init__(self):
        self.base_url = "https://api.github.com"
        self.session = requests.Session()
        # GitHub API doesn't require auth for basic searches, but rate limited

    def search_repositories(self, query: str, language: str = None,
                          sort: str = "stars", limit: int = 5) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        search_query = query
        if language:
            search_query += f" language:{language}"

        params = {
            "q": search_query,
            "sort": sort,
            "order": "desc",
            "per_page": limit
        }

        try:
            response = self.session.get(f"{self.base_url}/search/repositories", params=params, timeout=8)
            response.raise_for_status()

            results = []
            for repo in response.json().get("items", []):
                results.append({
                    "name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "stars": repo["stargazers_count"],
                    "language": repo.get("language"),
                    "url": repo["html_url"],
                    "topics": repo.get("topics", [])
                })
            return results

        except Exception as e:
            print(f"GitHub search error: {e}")
            return []

    def get_repository_structure(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository file structure"""
        try:
            response = self.session.get(f"{self.base_url}/repos/{owner}/{repo}/contents", timeout=8)
            response.raise_for_status()

            structure = []
            for item in response.json():
                if item["type"] == "file":
                    structure.append({
                        "name": item["name"],
                        "type": "file",
                        "path": item["path"]
                    })
                elif item["type"] == "dir":
                    structure.append({
                        "name": item["name"],
                        "type": "directory",
                        "path": item["path"]
                    })

            return {"structure": structure}

        except Exception as e:
            print(f"GitHub repo structure error: {e}")
            return {"structure": []}