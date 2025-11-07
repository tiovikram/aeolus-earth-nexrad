#!/usr/bin/env python3
"""
Fallback natural language parser for NEXRAD queries when Claude API is not available.
Uses simple regex patterns to extract stations and times.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Simple version compatible with the main parser."""
    stations: List[str]
    start_time: datetime
    end_time: datetime
    raw_query: str = ""
    confidence: float = 0.0
    additional_params: Dict = None


class FallbackNaturalLanguageParser:
    """Simple regex-based parser for common NEXRAD query patterns."""
    
    def __init__(self):
        """Initialize the fallback parser."""
        self.common_stations = [
            'KABR', 'KPDT', 'KYUX', 'KSGF', 'KRTX', 'KCXX', 'PGUA',
            'KBBX', 'KBGM', 'KBIS', 'KBLX', 'KBMX', 'KCAE', 'KCBW'
        ]
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse natural language query using simple regex patterns."""
        
        logger.info(f"Using fallback parser for: '{query}'")
        
        # Extract stations
        stations = self._extract_stations(query)
        if not stations:
            # Try to extract from common patterns
            if 'KABR' in query.upper() or 'Aberdeen' in query:
                stations = ['KABR']
            elif 'KPDT' in query.upper() or 'Portland' in query:
                stations = ['KPDT'] 
            elif 'KYUX' in query.upper():
                stations = ['KYUX']
            else:
                stations = ['KABR']  # Default fallback
        
        # Extract time information
        start_time, end_time = self._extract_times(query)
        
        confidence = 0.7 if stations and start_time and end_time else 0.3
        
        return ParsedQuery(
            stations=stations,
            start_time=start_time,
            end_time=end_time,
            raw_query=query,
            confidence=confidence
        )
    
    def _extract_stations(self, query: str) -> List[str]:
        """Extract station codes from query."""
        stations = []
        
        # Look for 4-letter station codes
        station_pattern = r'\b[K][A-Z]{3}\b'
        matches = re.findall(station_pattern, query.upper())
        
        for match in matches:
            if match in self.common_stations:
                stations.append(match)
        
        # Look for comma-separated lists
        if ',' in query:
            parts = query.split(',')
            for part in parts:
                station_match = re.search(station_pattern, part.upper())
                if station_match and station_match.group() in self.common_stations:
                    if station_match.group() not in stations:
                        stations.append(station_match.group())
        
        return stations
    
    def _extract_times(self, query: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract start and end times from query."""
        
        now = datetime.now()
        
        # Handle relative times
        if 'yesterday' in query.lower():
            base_date = now - timedelta(days=1)
            
            # Look for specific times
            if '2pm' in query.lower() or '2 pm' in query.lower() or '14:00' in query:
                start_time = base_date.replace(hour=14, minute=0, second=0, microsecond=0)
            else:
                start_time = base_date.replace(hour=12, minute=0, second=0, microsecond=0)
            
            if '6pm' in query.lower() or '6 pm' in query.lower() or '18:00' in query:
                end_time = base_date.replace(hour=18, minute=0, second=0, microsecond=0)
            else:
                end_time = start_time + timedelta(hours=4)  # 4-hour default
        
        elif 'today' in query.lower():
            base_date = now
            
            if 'morning' in query.lower():
                start_time = base_date.replace(hour=6, minute=0, second=0, microsecond=0)
                end_time = base_date.replace(hour=12, minute=0, second=0, microsecond=0)
            elif 'afternoon' in query.lower():
                start_time = base_date.replace(hour=12, minute=0, second=0, microsecond=0)
                end_time = base_date.replace(hour=18, minute=0, second=0, microsecond=0)
            else:
                start_time = base_date.replace(hour=12, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(hours=4)
        
        elif 'last' in query.lower():
            if 'hour' in query.lower():
                # Extract number of hours
                hour_match = re.search(r'last (\d+) hours?', query.lower())
                if hour_match:
                    hours = int(hour_match.group(1))
                    end_time = now - timedelta(minutes=30)  # 30 minutes ago for recent data
                    start_time = end_time - timedelta(hours=hours)
                else:
                    # Default to last 3 hours
                    end_time = now - timedelta(minutes=30)
                    start_time = end_time - timedelta(hours=3)
            
            elif 'night' in query.lower():
                yesterday = now - timedelta(days=1)
                start_time = yesterday.replace(hour=18, minute=0, second=0, microsecond=0)
                end_time = yesterday.replace(hour=23, minute=59, second=0, microsecond=0)
            else:
                # Generic "last" - assume last 6 hours
                end_time = now - timedelta(hours=1)
                start_time = end_time - timedelta(hours=6)
        
        else:
            # Look for specific date patterns like 2024-01-15
            date_pattern = r'(\d{4}-\d{1,2}-\d{1,2})'
            date_match = re.search(date_pattern, query)
            
            if date_match:
                try:
                    base_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    
                    # Look for time ranges
                    if '6am' in query.lower() or '6 am' in query.lower():
                        start_time = base_date.replace(hour=6, minute=0)
                    else:
                        start_time = base_date.replace(hour=12, minute=0)
                    
                    if 'noon' in query.lower():
                        end_time = base_date.replace(hour=12, minute=0)
                    elif '6pm' in query.lower() or '6 pm' in query.lower():
                        end_time = base_date.replace(hour=18, minute=0)
                    else:
                        end_time = start_time + timedelta(hours=6)
                        
                except ValueError:
                    start_time = None
                    end_time = None
            else:
                # Default fallback - yesterday afternoon
                yesterday = now - timedelta(days=1)
                start_time = yesterday.replace(hour=14, minute=0, second=0, microsecond=0)
                end_time = yesterday.replace(hour=18, minute=0, second=0, microsecond=0)
        
        return start_time, end_time


def test_fallback_parser():
    """Test the fallback parser with sample queries."""
    
    parser = FallbackNaturalLanguageParser()
    
    test_queries = [
        "Process KABR radar data from yesterday 2PM to 6PM",
        "Get KPDT data for the last 3 hours",
        "I need KYUX weather radar from today morning",
        "Process KABR, KPDT, and KYUX stations from last night",
        "Get data from KSGF for yesterday afternoon"
    ]
    
    for query in test_queries:
        try:
            result = parser.parse_query(query)
            print(f"Query: {query}")
            print(f"  Stations: {result.stations}")
            print(f"  Times: {result.start_time} to {result.end_time}")
            print(f"  Confidence: {result.confidence:.2f}")
            print()
        except Exception as e:
            print(f"Failed to parse '{query}': {e}")


if __name__ == '__main__':
    test_fallback_parser()