# CLAUDE.md

## Date Information Sources

### Current Date Source
I get the current date from the system environment information provided to me:
- **Current date**: September 26, 2025
- **Source**: System environment variable `Today's date: 2025-09-26`

### Local Data Alternative
You have data files in the `data/` folder that could be used for date information:
- `es_1D_data.csv` (207 KB)
- `es_1min_data_2015_2025.csv` (358 MB)

### Current Code Analysis
Looking at `main.py:12-17`, your code currently reads from:
```python
directorio = '../DATA'
nombre_fichero = 'export_es_2015_formatted.csv'
```

However, this points to a parent directory `../DATA` which may not exist, while you have actual data in the local `data/` folder.

### Recommendation
To use the local data folder instead of external date sources:
1. Change `directorio = '../DATA'` to `directorio = './data'`
2. Update `nombre_fichero` to match your actual files:
   - `'es_1D_data.csv'` or
   - `'es_1min_data_2015_2025.csv'`

This would make your code use the local data files for all date/time information rather than relying on external sources.