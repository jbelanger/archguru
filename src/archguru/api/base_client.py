"""
Base API client with common functionality
"""
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from ..core.constants import API_TIMEOUT

class BaseAPIClient(ABC):
    """Base class for API clients with common functionality"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = API_TIMEOUT
        
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        try:
            if headers:
                self.session.headers.update(headers)
                
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"{self.__class__.__name__} request error: {e}")
            return {}
        except Exception as e:
            print(f"{self.__class__.__name__} unexpected error: {e}")
            return {}
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> list:
        """Abstract search method to be implemented by subclasses"""
        pass