#!/usr/bin/env python3
"""
Natural Language Query Parser for NEXRAD Pipeline using Claude API.
Parses user queries like "Process KABR radar data from yesterday 2PM to 6PM" into structured parameters.
"""

import os
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a parsed natural language query."""
    stations: List[str]
    start_time: datetime
    end_time: datetime
    output_path: Optional[str] = None
    additional_params: Dict = None
    raw_query: str = ""
    confidence: float = 0.0


class NaturalLanguageParser:
    """Parser that uses Claude API to extract NEXRAD parameters from natural language."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the parser with Claude API key."""
        
        # Load .env file if available
        if DOTENV_AVAILABLE:
            # Try to load .env from current directory and parent directory
            load_dotenv()  # Current directory
            load_dotenv(dotenv_path=os.path.join('..', '.env'))  # Parent directory
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        if not self.api_key or self.api_key == 'your_api_key_here':
            logger.warning(
                "ANTHROPIC_API_KEY not set or has placeholder value. "
                "Will use fallback parser instead of Claude API."
            )
            self.use_fallback = True
            return
        else:
            self.use_fallback = False
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Common radar stations for context
        self.common_stations = [
            'KABR', 'KPDT', 'KYUX', 'KSGF', 'KRTX', 'KCXX', 'PGUA',
            'KBBX', 'KBGM', 'KBIS', 'KBLX', 'KBMX', 'KCAE', 'KCBW'
        ]
    
    def create_parsing_prompt(self, query: str) -> str:
        """Create a prompt for Claude to parse the natural language query."""
        
        current_time = datetime.now()
        
        prompt = f"""
You are a specialized parser for NEXRAD weather radar data queries. Parse the following natural language query into structured parameters for radar data processing.

Current date/time: {current_time.strftime('%Y-%m-%d %H:%M UTC')}

QUERY: "{query}"

Extract the following information and return ONLY a valid JSON object:

{{
    "stations": ["list of 4-letter radar station codes like KABR, KPDT"],
    "start_time": "YYYY-MM-DD-HH:MM format in UTC",
    "end_time": "YYYY-MM-DD-HH:MM format in UTC", 
    "confidence": 0.95,
    "reasoning": "Brief explanation of parsing decisions"
}}

PARSING RULES:
1. STATIONS: Extract 4-letter codes (KABR, KPDT, etc.). If locations mentioned, map to nearest station codes.
2. TIME PARSING:
   - "yesterday" = {(current_time - timedelta(days=1)).strftime('%Y-%m-%d')}
   - "today" = {current_time.strftime('%Y-%m-%d')}  
   - "2PM" = "14:00", "6AM" = "06:00"
   - "last 6 hours" = start 6 hours ago, end now
   - If only start time given, assume 4-hour duration
   - If no time zone specified, assume UTC
3. CONFIDENCE: 0.0-1.0 based on clarity of query

COMMON STATIONS:
{', '.join(self.common_stations[:20])}

EXAMPLES:
"Process KABR radar data from yesterday 2PM to 6PM" →
{{"stations": ["KABR"], "start_time": "{(current_time - timedelta(days=1)).strftime('%Y-%m-%d')}-14:00", "end_time": "{(current_time - timedelta(days=1)).strftime('%Y-%m-%d')}-18:00", "confidence": 0.9}}

"Get KPDT and KYUX data for last 3 hours" →
{{"stations": ["KPDT", "KYUX"], "start_time": "{(current_time - timedelta(hours=3)).strftime('%Y-%m-%d-%H:%M')}", "end_time": "{current_time.strftime('%Y-%m-%d-%H:%M')}", "confidence": 0.85}}

Return ONLY the JSON object, no additional text.
"""
        return prompt
    
    def parse_query(self, query: str) -> ParsedQuery:
        """Parse a natural language query using Claude API or fallback parser."""
        
        # Use fallback parser if Claude API not available
        if getattr(self, 'use_fallback', False):
            try:
                from fallback_parser import FallbackNaturalLanguageParser
                fallback = FallbackNaturalLanguageParser()
                return fallback.parse_query(query)
            except ImportError:
                raise ValueError("Fallback parser not available and Claude API not configured")
        
        try:
            prompt = self.create_parsing_prompt(query)
            
            logger.info(f"Parsing query with Claude API: '{query}'")
            
            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                temperature=0.1,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            response_text = message.content[0].text.strip()
            logger.debug(f"Claude response: {response_text}")
            
            # Parse JSON response
            try:
                parsed_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Try to extract JSON from response if wrapped in text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from Claude response: {e}")
            
            # Validate and convert to ParsedQuery
            stations = parsed_data.get('stations', [])
            start_time_str = parsed_data.get('start_time', '')
            end_time_str = parsed_data.get('end_time', '')
            confidence = parsed_data.get('confidence', 0.0)
            
            # Convert time strings to datetime objects
            start_time = self._parse_time_string(start_time_str)
            end_time = self._parse_time_string(end_time_str)
            
            # Validate results
            if not stations:
                raise ValueError("No radar stations found in query")
            if not start_time or not end_time:
                raise ValueError("Could not parse start/end times from query")
            if start_time >= end_time:
                raise ValueError("Start time must be before end time")
            
            logger.info(f"Successfully parsed: {len(stations)} stations, "
                       f"{start_time} to {end_time} (confidence: {confidence:.2f})")
            
            return ParsedQuery(
                stations=stations,
                start_time=start_time,
                end_time=end_time,
                raw_query=query,
                confidence=confidence,
                additional_params=parsed_data
            )
            
        except Exception as e:
            logger.error(f"Claude API failed: {e}")
            
            # If Claude API fails, try fallback parser
            if "credit balance" in str(e) or "rate limit" in str(e) or "400" in str(e):
                logger.info("Claude API unavailable, trying fallback parser...")
                try:
                    from fallback_parser import FallbackNaturalLanguageParser
                    fallback = FallbackNaturalLanguageParser()
                    result = fallback.parse_query(query)
                    logger.info(f"Fallback parser succeeded with confidence: {result.confidence:.2f}")
                    return result
                except ImportError:
                    raise ValueError("Fallback parser not available and Claude API failed")
                except Exception as fallback_error:
                    logger.error(f"Fallback parser also failed: {fallback_error}")
            
            raise ValueError(f"Query parsing failed: {e}")
    
    def _parse_time_string(self, time_str: str) -> Optional[datetime]:
        """Parse time string in YYYY-MM-DD-HH:MM format to datetime."""
        if not time_str:
            return None
            
        try:
            return datetime.strptime(time_str, '%Y-%m-%d-%H:%M')
        except ValueError:
            try:
                # Try with seconds
                return datetime.strptime(time_str, '%Y-%m-%d-%H:%M:%S')
            except ValueError:
                logger.error(f"Could not parse time string: {time_str}")
                return None
    
    def validate_stations(self, stations: List[str]) -> Tuple[List[str], List[str]]:
        """Validate station codes and return valid/invalid lists."""
        
        valid_stations = []
        invalid_stations = []
        
        for station in stations:
            station = station.upper().strip()
            if re.match(r'^[A-Z]{4}$', station):
                valid_stations.append(station)
            else:
                invalid_stations.append(station)
        
        return valid_stations, invalid_stations


def test_parser():
    """Test the natural language parser with sample queries."""
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("ANTHROPIC_API_KEY not set - skipping parser tests")
        return
    
    parser = NaturalLanguageParser()
    
    test_queries = [
        "Process KABR radar data from yesterday 2PM to 6PM",
        "Get KPDT and KYUX data for the last 3 hours", 
        "I need KSGF weather radar from today 6AM to noon",
        "Analyze PGUA station data from 2024-01-15 morning",
        "Process multiple stations KABR,KPDT,KYUX from last night"
    ]
    
    for query in test_queries:
        try:
            result = parser.parse_query(query)
            logger.info(f"Query: {query}")
            logger.info(f"  Stations: {result.stations}")
            logger.info(f"  Time: {result.start_time} to {result.end_time}")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            print()
        except Exception as e:
            logger.error(f"Failed to parse '{query}': {e}")


if __name__ == '__main__':
    test_parser()