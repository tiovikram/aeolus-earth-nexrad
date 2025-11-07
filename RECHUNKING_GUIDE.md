# xarray_beam Rechunking Guide for NEXRAD Pipeline

The `run_nlp_pipeline_xbeam.py` pipeline includes comprehensive rechunking functionality to optimize data processing and memory usage for NEXRAD datasets.

## Rechunking Strategies

### 1. **Auto (Default)**
```bash
python run_nlp_pipeline_xbeam.py --query "Process KABR data" --rechunk_size "auto"
```
- **Behavior**: No explicit rechunking applied
- **Use case**: Small datasets or when original chunking is optimal
- **Performance**: Minimal overhead

### 2. **Optimal Chunking**
```bash
python run_nlp_pipeline_xbeam.py --query "Process KABR data" --rechunk_size "optimal"
```
- **Behavior**: Intelligent chunking based on NEXRAD data characteristics
- **Algorithm**:
  - Time dimension: Max 10 time steps per chunk
  - Spatial dimensions (range, azimuth, x, y): 
    - ≤100 elements: Keep unchunked
    - ≤1000 elements: Split in half
    - >1000 elements: Use 500-element chunks
- **Use case**: General-purpose processing with unknown data dimensions

### 3. **Numeric Chunk Size**
```bash
python run_nlp_pipeline_xbeam.py --query "Process KABR data" --rechunk_size "10"
```
- **Behavior**: Apply chunk size of 10 to the time dimension
- **Use case**: Simple temporal chunking for time-series analysis

### 4. **Dimension-Specific Chunking**
```bash
python run_nlp_pipeline_xbeam.py --query "Process KABR data" --rechunk_size "time:10,range:500,azimuth:360"
```
- **Behavior**: Apply different chunk sizes to different dimensions
- **Format**: `dimension:size,dimension:size`
- **Use case**: Fine-tuned optimization for specific processing workflows

## Chunking Implementation

### ProcessDataset DoFn Enhancement

```python
class ProcessDataset(beam.DoFn):
    def __init__(self, rechunk_size: str = 'auto'):
        self.rechunk_size = rechunk_size
    
    def _rechunk_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply rechunking strategy to dataset."""
        
        # Parse rechunk specifications
        if self.rechunk_size.isdigit():
            chunk_dict = {'time': int(self.rechunk_size)}
        elif self.rechunk_size == 'optimal':
            chunk_dict = self._get_optimal_chunks(ds)
        elif ',' in self.rechunk_size:
            # Parse "dim:size,dim:size" format
            chunk_dict = self._parse_chunk_spec(self.rechunk_size)
            
        # Apply only to existing dimensions
        valid_chunks = {dim: size for dim, size in chunk_dict.items() 
                       if dim in ds.dims}
        
        return ds.chunk(valid_chunks) if valid_chunks else ds
```

### Optimal Chunking Algorithm

```python
def _get_optimal_chunks(self, ds: xr.Dataset) -> Dict[str, int]:
    """Calculate optimal chunk sizes for NEXRAD data."""
    optimal_chunks = {}
    
    # Time dimension: small chunks for temporal processing
    if 'time' in ds.dims:
        time_size = ds.sizes['time']
        optimal_chunks['time'] = min(10, time_size)
    
    # Spatial dimensions: larger chunks for spatial operations  
    spatial_dims = ['range', 'azimuth', 'x', 'y', 'lat', 'lon']
    for dim in spatial_dims:
        if dim in ds.dims:
            dim_size = ds.sizes[dim]
            if dim_size <= 100:
                optimal_chunks[dim] = dim_size      # Keep small unchunked
            elif dim_size <= 1000:
                optimal_chunks[dim] = dim_size // 2  # Split large in half
            else:
                optimal_chunks[dim] = 500           # Fixed chunks for very large
                
    return optimal_chunks
```

## Chunk Information and Debugging

### Dataset Attributes
Processed datasets include chunk information in attributes:
```python
ds.attrs['chunk_info']  # String representation of chunk structure
```

### Chunk Inspection
```python
import xarray as xr
ds = xr.open_zarr('out/KABR/20241101_1201.zarr')

# Check chunk information
print(f"Chunk info: {ds.attrs.get('chunk_info', 'Not available')}")

# Inspect variable chunking
for var_name, var in ds.data_vars.items():
    if hasattr(var.data, 'chunks'):
        print(f'{var_name}: chunks={var.data.chunks}')
```

## Performance Considerations

### Memory Usage
- **Smaller chunks**: Lower memory usage per operation, higher overhead
- **Larger chunks**: Higher memory usage, better computational efficiency
- **Optimal chunks**: Balance between memory and performance

### Processing Patterns
- **Temporal analysis**: Use small time chunks (`time:1` or `time:10`)
- **Spatial operations**: Use larger spatial chunks (`range:500,azimuth:180`)
- **I/O optimization**: Match chunk sizes to storage block sizes

### Network and Storage
- **S3 output**: Larger chunks reduce number of objects created
- **Local output**: Chunk size affects individual file sizes in Zarr arrays

## Examples

### Time Series Analysis
```bash
python run_nlp_pipeline_xbeam.py \
  --query "Process KABR data from last 6 hours" \
  --rechunk_size "time:1"  # One time step per chunk
```

### Spatial Processing
```bash
python run_nlp_pipeline_xbeam.py \
  --query "Process multiple stations" \
  --rechunk_size "range:1000,azimuth:360"  # Optimize for spatial ops
```

### Large Dataset Processing
```bash
python run_nlp_pipeline_xbeam.py \
  --query "Process KABR data from November 1st 2024" \
  --rechunk_size "optimal" \
  --s3_output_bucket my-nexrad-results
```

### Custom Multi-Dimensional
```bash
python run_nlp_pipeline_xbeam.py \
  --query "Process KABR and KPDT stations" \
  --rechunk_size "time:5,range:250,azimuth:90"
```

## Error Handling

### Rechunking Failures
- Pipeline continues with original chunking if rechunking fails
- Warnings logged for debugging
- No data loss on rechunking errors

### Invalid Dimensions
- Only dimensions that exist in dataset are rechunked
- Invalid dimension specifications are ignored
- Graceful fallback to valid chunks only

## Integration with Apache Beam

The rechunking functionality is seamlessly integrated with Apache Beam processing:
1. **File Loading**: Creates initial datasets with default chunking
2. **Processing Stage**: Applies rechunking strategy during dataset processing
3. **Output Stage**: Preserves chunk structure in saved Zarr datasets
4. **Pipeline Reporting**: Includes chunking information in result metadata

This provides fine-grained control over data chunking throughout the entire xarray_beam pipeline while maintaining the distributed processing benefits of Apache Beam.