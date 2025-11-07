# Agent Handoff Documentation

## Current Status

Working on fixing an inconsistency in the `src/run_nlp_pipeline_xbeam.py` file related to station discovery logic.

## Problem Identified

The user correctly identified that there's an inconsistency in the station discovery implementation:

1. **Line 50**: Still has hardcoded stations: `self.stations = ['KABR', 'KPDT', 'KYUX', 'KSGF', 'KRTX', 'PGUA']`
2. **Lines 57-59**: Code loops through `self.stations` to find stations mentioned in query
3. **Line 63**: Only sets `stations = None` if NO hardcoded stations are found

## Current Implementation Issues

The current logic means dynamic discovery only works if the query doesn't mention any of the 6 hardcoded station names. This is inconsistent with the goal of true dynamic discovery.

## What Needs to be Fixed

The user's original request was to have dynamic station discovery that pulls from the S3 bucket structure rather than hardcoded station lists. The current implementation has:

- ✅ `discover_available_stations()` function (lines 459-490) that properly discovers stations from S3
- ✅ `find_nexrad_files()` updated to handle `stations: Optional[List[str]]` (line 493)
- ❌ Still using hardcoded station list in parsing logic (lines 50, 57-59)

## User's Key Point

The user reminded me that we're using an LLM/natural language approach, not regex parsing. The current simple parser is just a fallback, and the real solution should leverage the fact that we have `discover_available_stations()` for true dynamic discovery.

## Files Modified

- `src/run_nlp_pipeline_xbeam.py` - Main pipeline file with xarray_beam integration

## Key Changes Already Made (Working)

1. **Dynamic Station Discovery Function** (lines 459-490):
   ```python
   def discover_available_stations(date: datetime) -> List[str]:
       """Discover all available radar stations for a given date from S3."""
   ```

2. **Updated find_nexrad_files** (line 493):
   ```python
   def find_nexrad_files(stations: Optional[List[str]], start_time: datetime, end_time: datetime, max_files_per_station: int = 3)
   ```

3. **Station discovery logic** (lines 497-499):
   ```python
   if stations is None:
       stations = discover_available_stations(start_time)
   ```

## What Still Needs to be Done

1. **Remove or fix the hardcoded station logic** in `SimpleNLParser.__init__()` and `parse_query()` method
2. **The inconsistency is in lines 50 and 57-59** - these still reference hardcoded stations
3. **The logic should be**: If no specific stations mentioned in query → set `stations = None` → trigger dynamic discovery

## Testing Status

- ✅ Dry run tests show the pipeline recognizes when `stations = None` 
- ✅ Dynamic discovery function is implemented and ready
- ✅ Pipeline handles both specific stations and dynamic discovery
- ❌ Still has hardcoded station list that needs removal

## User Requirements Summary

1. Default directory is `data/` (unless query is passed)
2. If query is passed, use NEXRAD S3 data source  
3. If no radar stations in query, discover ALL stations from S3 bucket structure dynamically
4. Default output is `out/`
5. Auto-detect S3 vs local output paths based on `s3://` prefix

## Current Working CLI

```bash
# Default local data processing
python src/run_nlp_pipeline_xbeam.py

# Dynamic station discovery (all stations)  
python src/run_nlp_pipeline_xbeam.py --query "Process radar data from November 1st 2024"

# Specific station
python src/run_nlp_pipeline_xbeam.py --query "Process KABR radar data from November 1st 2024"
```

## Next Steps for Continuing Agent

1. Fix the hardcoded station inconsistency in `SimpleNLParser` class
2. Remove `self.stations` hardcoded list (line 50)
3. Update station extraction logic (lines 57-59) to properly handle "no specific stations mentioned" case
4. Test that dynamic discovery works for both specific and general queries
5. The goal is true dynamic discovery from S3 bucket structure, not hardcoded station lists

## Important Note

The user emphasized that we should be leveraging the natural language approach and dynamic discovery, not trying to implement complex regex parsing. The `discover_available_stations()` function is the key - it should be used when no specific stations are mentioned in the query.