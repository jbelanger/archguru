"""
StackOverflow API client for technical research
"""
import requests
from typing import List, Dict, Any
from ..core.config import Config


class StackOverflowClient:
    """Client for StackOverflow API research"""

    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3"
        self.session = requests.Session()

    def search_questions(self, query: str, tags: List[str] = None,
                        sort: str = "votes", limit: int = 5) -> List[Dict[str, Any]]:
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

        try:
            response = self.session.get(f"{self.base_url}/search", params=params, timeout=8)
            response.raise_for_status()

            results = []
            data = response.json()
            for question in data.get("items", []):
                results.append({
                    "title": question.get("title", ""),
                    "score": question.get("score", 0),
                    "view_count": question.get("view_count", 0),
                    "answer_count": question.get("answer_count", 0),
                    "tags": question.get("tags", []),
                    "url": question.get("link", ""),
                    "is_answered": question.get("is_answered", False)
                })

            return results

        except Exception as e:
            print(f"StackOverflow search error: {e}")
            return []

    def get_popular_tags(self, related_to: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popular tags, optionally filtered"""
        params = {
            "order": "desc",
            "sort": "popular",
            "site": "stackoverflow",
            "pagesize": limit
        }

        if related_to:
            params["inname"] = related_to

        try:
            response = self.session.get(f"{self.base_url}/tags", params=params, timeout=8)
            response.raise_for_status()

            results = []
            data = response.json()
            for tag in data.get("items", []):
                results.append({
                    "name": tag.get("name", ""),
                    "count": tag.get("count", 0)
                })

            return results

        except Exception as e:
            print(f"StackOverflow tags error: {e}")
            return []