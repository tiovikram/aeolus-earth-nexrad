## Building a NEXRAD preprocessing pipeline

Important links:
- [EarthMover blog on radar](https://earthmover.io/blog/from-files-to-datasets-fm-301-and-the-future-of-radar-interoperability) -- there are some more useful blogs on their site as well for radar.
- [`xarray-beam` docs](https://xarray-beam.readthedocs.io/en/latest/index.html)
- [NEXRAD on AWS](https://registry.opendata.aws/noaa-nexrad/)


The purpose of this task is to build a preprocessing pipeline for NEXRAD data. 
NEXRAD is a weather radar system that provides various radar products. The data is all stored in S3, but it's in a binary format and is scattered across many files. 
In order to run a forecasting model on radar data, we need to process it into a format that can be used by the model. One of the big things here is converting from the NEXRAD grid (polar) to the model grid (cartesian). 

A cartesian grid consists of (x, y, z) coordinates. In Zarr terms, for a 100x100x10 grid, you can think of `x` being an array like [1, 2, 3, ..., 100], and `lat(x)` being an array like [39.9, 39.91, 39.92, ..., 40.0]. The same goes for `y` and `lon(y)`. `z` is a simple array like [0, 1, 2, ..., 9] and `height(z)` is an array like [0, 10, 200, ..., 10000] (in units of meters). `lat(x)`, `lon(y)`, and `height(z)` map from the model grid to the spatial grid. 

[This blog](https://earthmover.io/blog/from-files-to-datasets-fm-301-and-the-future-of-radar-interoperability) explains the structure of NEXRAD data and how packages like `xradar` are used to load it into an `xarray.DataTree` and visualize it. This is a good starting point, but we also need to perform a regrid. 

[`xarray-beam`](https://xarray-beam.readthedocs.io/en/latest/index.html) is a library for writing Apache Beam pipelines consisting of xarray Dataset objects. Beam is a programming model for defining and executing both batch and streaming data processing pipelines, and the xarray-beam library provides a set of primitives for working with xarray objects in a Beam pipeline (as you'll see in the docs). 

Your task is to implement a function `process_nexrad_data` which can do the following:
- Load in NEXRAD data from data/ into an `xarray.DataTree`. 
- Regrid the NEXRAD data to the model grid (Q: *how do we define a model grid?*) using `xarray-beam`.
- Rechunk the data into an efficient format for the model. You don't want to hardcode the chunk size here.
- Write the processed data to a Zarr store in out/. 
- **Bonus**: add a QC/health check framework for the data. Things we care about here: 
    - Coverage and completeness metrics
    - Physical bounds and value range validation
    - Signal quality assessment



It's your job to define what the function signature should be as well as the implementation. You should also write tests/visualizations to convince yourself (and us) that the code is working as expected.

This task is designed to be pretty vague/open-ended and you are not expected to understand everything from the get-go. What we care about is how fast you can ramp up and get something working.

## Setup

### Environment Variables

The pipeline uses the Claude API for natural language query parsing. You need to set up your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Alternatively, create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

**Note**: If the Claude API is unavailable (e.g., no API key or insufficient credits), the pipeline will fall back to default parameters (all stations from November 1st, 2024, 00:00-06:00).

## Usage

The pipeline can be run with natural language queries:

```bash
# Process specific station data
python src/run_nlp_pipeline_xbeam.py --query "Process KABR radar data from November 1st 2024"

# Process data with custom limits
python src/run_nlp_pipeline_xbeam.py --query "Process radar data from yesterday" --max_files_per_station 5 --max_stations 20
```

### Important Configuration Parameters

- **`--max_stations`** (default: 10): When no specific stations are mentioned in the query, the pipeline will discover all available stations but only process the first N stations. This is set to 10 by default for efficiency. To process all discovered stations, increase this limit (e.g., `--max_stations 200`).
- **`--max_files_per_station`** (default: 3): Maximum number of files to process per station.
- **`--output`** (default: out/): Output directory for processed data. Can be a local path or S3 path (e.g., `s3://my-bucket/results`).

### Examples

```bash
# Process all available stations (may take longer)
python src/run_nlp_pipeline_xbeam.py --query "Process all radar data from November 1st 2024" --max_stations 500

# Process local data from data/ directory
python src/run_nlp_pipeline_xbeam.py

# Save output to S3
python src/run_nlp_pipeline_xbeam.py --query "Process KABR data" --output s3://my-bucket/nexrad-output
``` 
