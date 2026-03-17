# Data ingestion module
# Aggregates public health telemetry from structured and unstructured sources

import json
from typing import Dict, Any, List
import requests
import feedparser

class DiseaseHarmonizer:
    # Class for handling multi-source data harmonization
    def __init__(self):
        # API endpoints
        self.disease_api = "https://disease.sh/v3/covid-19/all"
        self.promed_rss = "https://promedmail.org/feed/"
        
    def fetch_global_stats(self) -> Dict[str, Any]:
        """Fetch global COVID statistics."""
        try:
            response = requests.get(self.disease_api, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def fetch_promed_alerts(self, limit: int = 5) -> List[Dict[str, str]]:
        """Parse ProMED clinical alerts from RSS."""
        try:
            feed = feedparser.parse(self.promed_rss)
            alerts = []
            for entry in feed.entries[:limit]:
                alerts.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": entry.summary,
                    "published": entry.published
                })
            return alerts
        except (AttributeError, KeyError) as e:
            return [{"error": str(e)}]

    def harmonize(self, disease_query: str = "COVID-19") -> Dict[str, Any]:
        # Unify structured data with unstructured alerts
        stats = self.fetch_global_stats()
        alerts = self.fetch_promed_alerts()
        
        payload = {
            "metadata": {
                "source": "Global Surveillance",
                "target": disease_query
            },
            "stats": stats,
            "alerts": alerts,
            "status": f"Complete for {disease_query}"
        }
        
        return payload

if __name__ == "__main__":
    h = DiseaseHarmonizer()
    print(json.dumps(h.harmonize(), indent=2))
