"""
Interactive batch runner for show_ODs_v2 analysis.

This script:
1. Prompts user to select a date
2. Shows available sequences for that date
3. Prompts user to select which sequences to process
4. Runs show_ODs analysis on selected files (without displaying plots)
5. Saves analysis results to the HDF5 'results/show_ODs_v2' group
6. Enables subsequent multishot_lib filtering to work with the analysis data

Configuration:
    Set SEQS and HDF_CONFIG below to skip interactive prompts and use those directly.
    
Usage:
    python run_show_ODs_batch.py
    
    Follow prompts to select date and sequences, or configure SEQS/HDF_CONFIG below.
"""

# =========================================================================
# CONFIGURATION - Set these to skip interactive prompts
# =========================================================================

# Optional: Set to specific sequence indices to skip selection prompt
# Example: SEQS = [26, 30]  or set to None to prompt user
SEQS = [33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47]  # type: list | None

# Optional: Set to specify date directly to skip date selection prompt
# Example: HDF_CONFIG = {'today': False, 'year': 2026, 'month': 4, 'day': 15}
# or: HDF_CONFIG = {'today': True}  to use today's date
HDF_CONFIG = {'today': False, 'year': 2026, 'month': 3, 'day': 24}  # type: dict | None

# =========================================================================


import h5py
import sys
import logging
import os
import glob
from pathlib import Path
import traceback
import getpass
import importlib.util
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from datetime import timedelta

# Set matplotlib backend BEFORE importing pyplot or show_ODs_v2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation

# =========================================================================
# PERFORMANCE TUNING
# =========================================================================
NUM_WORKERS = max(1, min(4, cpu_count() - 1))  # Use up to 4 workers to avoid HDF5 file locking
SKIP_EXISTING = False  # Always re-analyze files (set to True to skip if results exist)
DISABLE_LOGGING = True  # Set to True to suppress INFO logging (speeds up output)
# =========================================================================

# Timing utilities
class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
    
    def format(self):
        if self.elapsed < 1:
            return f"{self.elapsed*1000:.0f}ms"
        elif self.elapsed < 60:
            return f"{self.elapsed:.1f}s"
        else:
            return str(timedelta(seconds=self.elapsed)).split('.')[0]

# Setup logging
if DISABLE_LOGGING:
    logging.disable(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2')

# Load show_ODs_v2
spec = importlib.util.spec_from_file_location(
    "show_ODs_v2",
    "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis/show_ODs_v2.py"
)
if spec and spec.loader:
    show_ODs_v2_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(show_ODs_v2_mod)
else:
    logger.error("Could not load show_ODs_v2 module")
    sys.exit(1)


def get_hdf_files_for_date(date_str, seq_indices=None, hdf_root=None):
    """
    Get all HDF5 shot files for a specific date and optional sequence indices.
    
    Parameters
    ----------
    date_str : str
        Date in format 'YYYY-MM-DD' (e.g., '2026-04-15')
    seq_indices : list, optional
        List of sequence indices (e.g., [26, 30]). If None, all sequences in date.
    hdf_root : str, optional
        Root directory containing HDF files. If None, uses NAS542_dataBEC2 with auto-detection of username.
    
    Returns
    -------
    dict
        Dictionary mapping sequence number to list of HDF5 files:
        {26: ['/path/to/0026/file1.h5', '/path/to/0026/file2.h5'], 30: [...]}
    """
    if hdf_root is None:
        username = getpass.getuser()
        hdf_root = f"/home/{username}/NAS542_dataBEC2"
    
    # Parse date
    parts = date_str.split('-')
    if len(parts) != 3:
        raise ValueError(f"Date must be in format YYYY-MM-DD, got: {date_str}")
    
    year, month, day = parts
    date_dir = os.path.join(hdf_root, year, month, day)
    
    if not os.path.exists(date_dir):
        logger.warning(f"Date directory does not exist: {date_dir}")
        return {}
    
    files_by_seq = {}
    
    # If seq_indices specified, search only those
    if seq_indices:
        search_dirs = [os.path.join(date_dir, f"{seq:04d}") for seq in seq_indices]
    else:
        # Find all sequence subdirectories
        search_dirs = glob.glob(os.path.join(date_dir, '[0-9][0-9][0-9][0-9]'))
    
    for seq_dir in sorted(search_dirs):
        if not os.path.isdir(seq_dir):
            continue
        
        seq_num = int(os.path.basename(seq_dir))
        
        # Find all .h5 files in this sequence directory
        hdf_files = sorted(glob.glob(os.path.join(seq_dir, '*.h5')))
        
        if hdf_files:
            files_by_seq[seq_num] = hdf_files
            logger.info(f"Found {len(hdf_files)} HDF5 files in sequence {seq_num:04d}")
    
    return files_by_seq


def process_file(hdf_file_path, consolidation=None, sequence_num=None):
    """
    Run show_ODs analysis on a single HDF5 file and save results.
    
    Parameters
    ----------
    hdf_file_path : str
        Path to HDF5 result file
    consolidation : dict, optional
        Consolidation tracking dict to accumulate results into DataFrame
    sequence_num : int, optional
        Sequence number for routing to correct consolidation
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        with h5py.File(hdf_file_path, 'r+') as h5file:
            # Check if already processed and skip if configured
            if SKIP_EXISTING:
                if 'results' in h5file and 'show_ODs_v2' in h5file['results']:
                    return True  # Already processed, count as success
            
            # Run analysis (show=False to skip plotting)
            dict_results = show_ODs_v2_mod.show_ODs(h5file, show=False, show_SVD=True)
            
            if dict_results is None or len(dict_results) == 0:
                logger.warning(f"  No results generated for {os.path.basename(hdf_file_path)}")
                return False
            
            # Create/get results group
            if 'results' not in h5file:
                h5file.create_group('results')
            
            results_group = h5file['results']
            
            # Create show_ODs_v2 group if not exists
            if 'show_ODs_v2' not in results_group:
                show_ODs_group = results_group.create_group('show_ODs_v2')
            else:
                show_ODs_group = results_group['show_ODs_v2']
                logger.info(f"  Overwriting existing show_ODs_v2 results")
            
            # Save all results to HDF5
            for key, value in dict_results.items():
                try:
                    # Handle numpy arrays and scalars
                    if isinstance(value, (int, float, str, bool, bytes)):
                        show_ODs_group.attrs[f"{key}"] = value
                    elif hasattr(value, '__len__') and not isinstance(value, str):
                        # Array-like data
                        if f"{key}" in show_ODs_group:
                            del show_ODs_group[f"{key}"]
                        show_ODs_group.create_dataset(f"{key}", data=value)
                    else:
                        # Try as attribute
                        show_ODs_group.attrs[f"{key}"] = value
                except Exception as e:
                    logger.debug(f"    Could not save {key}: {e}")
            
            logger.info(f"  ✓ Saved {len(dict_results)} results")
            
            # Add to consolidation DataFrame if provided
            if consolidation is not None and sequence_num is not None:
                try:
                    row_data = dict(dict_results)
                    
                    # Add shot filename for identification
                    row_data['shot_name'] = os.path.splitext(os.path.basename(hdf_file_path))[0]
                    
                    # Extract and add root-level h5 attributes
                    for key, value in h5file.attrs.items():
                        attr_key = f"h5_attr_{key}"
                        try:
                            row_data[attr_key] = value
                        except Exception as e:
                            logger.debug(f"    Could not add h5 attribute {key}: {e}")
                    
                    # Extract and add globals group attributes
                    if 'globals' in h5file:
                        globals_group = h5file['globals']
                        for key, value in globals_group.attrs.items():
                            attr_key = f"globals_attr_{key}"
                            try:
                                row_data[attr_key] = value
                            except Exception as e:
                                logger.debug(f"    Could not add globals attribute {key}: {e}")
                    
                    # Add to correct sequence's consolidation
                    if sequence_num in consolidation:
                        consolidation[sequence_num]['rows'].append(row_data)
                except Exception as e:
                    logger.debug(f"  Could not add to consolidation: {e}")
            
            return True
    
    except Exception as e:
        logger.error(f"Error processing {hdf_file_path}: {e}")
        traceback.print_exc()
        return False


def process_file_wrapper(args):
    """
    Wrapper for parallel processing of files.
    Returns filename and success status for progress tracking.
    """
    hdf_file_path, = args
    filename = os.path.basename(hdf_file_path)
    success = False
    try:
        success = process_file(hdf_file_path)
    except Exception as e:
        logger.debug(f"Error in wrapper for {filename}: {e}")
    return filename, success


def get_date_list(args):
    """
    Parse command line arguments to get list of dates.
    
    Parameters
    ----------
    args : list
        Command line arguments (sys.argv[1:])
    
    Returns
    -------
    list
        List of date strings in format 'YYYY-MM-DD'
    """
    from datetime import datetime, timedelta
    
    if not args:
        # Default: today
        return [datetime.now().strftime('%Y-%m-%d')]
    
    dates = []
    for arg in args:
        try:
            # Try parsing as single date
            dt = datetime.strptime(arg, '%Y-%m-%d')
            dates.append(dt.strftime('%Y-%m-%d'))
        except ValueError:
            logger.warning(f"Could not parse date: {arg} (use YYYY-MM-DD format)")
    
    return dates


def prompt_for_date(hdf_root=None):
    """
    Prompt user to select a date or enter custom date.
    
    Parameters
    ----------
    hdf_root : str, optional
        Root directory containing HDF files
    
    Returns
    -------
    str or None
        Selected date in format 'YYYY-MM-DD', or None if cancelled
    """
    from datetime import datetime, timedelta
    
    # Check if HDF_CONFIG is set to use that directly
    if HDF_CONFIG is not None:
        if HDF_CONFIG.get('today', False):
            date_str = datetime.now().strftime('%Y-%m-%d')
            print(f"\n✓ Using HDF_CONFIG: today ({date_str})")
            return date_str
        else:
            year = HDF_CONFIG.get('year')
            month = HDF_CONFIG.get('month')
            day = HDF_CONFIG.get('day')
            if all([year, month, day]):
                date_str = f"{year:04d}-{month:02d}-{day:02d}"
                print(f"\n✓ Using HDF_CONFIG: {date_str}")
                return date_str
    
    print("\n" + "="*60)
    print("SELECT DATE FOR ANALYSIS")
    print("="*60)
    
    # Try to load waterfall_config_v2 default
    waterfall_date = None
    try:
        sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2')
        import waterfall_config_v2
        if hasattr(waterfall_config_v2, 'HDF_CONFIG'):
            hdf_cfg = waterfall_config_v2.HDF_CONFIG
            if not hdf_cfg.get('today', False):
                year = hdf_cfg.get('year')
                month = hdf_cfg.get('month')
                day = hdf_cfg.get('day')
                if all([year, month, day]):
                    waterfall_date = f"{year:04d}-{month:02d}-{day:02d}"
    except Exception as e:
        pass
    
    # Offer recent dates
    today = datetime.now()
    dates_to_offer = [today - timedelta(days=i) for i in range(7)]
    
    print("\nRecent dates:")
    for i, dt in enumerate(dates_to_offer, 1):
        date_str = dt.strftime('%Y-%m-%d')
        marker = " [waterfall_config_v2]" if date_str == waterfall_date else ""
        print(f"  {i}. {date_str}{marker}")
    
    print(f"  0. Enter custom date (YYYY-MM-DD format)")
    print(f"  q. Quit")
    
    while True:
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice == '0':
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD")
                continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(dates_to_offer):
                return dates_to_offer[idx].strftime('%Y-%m-%d')
            else:
                print(f"Please select 0-{len(dates_to_offer)} or 'q'")
        except ValueError:
            print("Invalid choice. Enter a number, 0, or 'q'")


def prompt_for_sequences(files_by_seq, date_str):
    """
    Prompt user to select which sequences to process.
    
    Parameters
    ----------
    files_by_seq : dict
        Dictionary mapping sequence number to list of HDF5 files
    date_str : str
        Date in YYYY-MM-DD format
    
    Returns
    -------
    dict or None
        Selected sequences as {seq_num: [files]}, or None if cancelled
    """
    if not files_by_seq:
        return None
    
    # Check if SEQS is configured to use that directly
    if SEQS is not None:
        selected = {seq: files_by_seq[seq] for seq in SEQS if seq in files_by_seq}
        if selected:
            print(f"\n✓ Using SEQS configuration: {SEQS}")
            print(f"  Found {len(selected)} sequence(s):")
            for seq in sorted(selected.keys()):
                print(f"    - seq_{seq:04d}  ({len(selected[seq])} files)")
            return selected
        else:
            print(f"\n✗ SEQS={SEQS} specified but not found in available sequences")
            print(f"  Available: {sorted(files_by_seq.keys())}")
            return None
    
    print("\n" + "="*60)
    print("SELECT SEQUENCES TO PROCESS")
    print("="*60)
    
    seq_list = sorted(files_by_seq.keys())
    
    print(f"\nFound {len(seq_list)} sequence(s):\n")
    for i, seq_num in enumerate(seq_list, 1):
        num_files = len(files_by_seq[seq_num])
        print(f"  {i}. seq_{seq_num:04d}  ({num_files} files)")
    
    print(f"\n  a. Process ALL sequences")
    print(f"  w. Use sequences from waterfall_config_v2.SEQS")
    print(f"  s. Enter sequence indices (e.g., '26 30' or '26,30')")
    print(f"  q. Quit (don't process anything)")
    
    while True:
        choice = input("\nSelect sequences (number, 'a', 'w', 's', or 'q'): ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice == 'a':
            return files_by_seq
        
        if choice == 'w':
            # Try to load from waterfall_config_v2
            try:
                sys.path.insert(0, '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/waterfall_v2')
                import waterfall_config_v2
                
                if hasattr(waterfall_config_v2, 'SEQS') and waterfall_config_v2.SEQS:
                    seqs = waterfall_config_v2.SEQS
                    print(f"\nUsing SEQS from waterfall_config_v2: {seqs}")
                    selected = {seq: files_by_seq[seq] for seq in seqs if seq in files_by_seq}
                    if selected:
                        print(f"✓ Selected {len(selected)} sequence(s):")
                        for seq in sorted(selected.keys()):
                            print(f"  - seq_{seq:04d}  ({len(selected[seq])} files)")
                        return selected
                    else:
                        print(f"✗ Could not find sequences {seqs} in available sequences")
                        print(f"Available: {seq_list}")
                else:
                    print("✗ waterfall_config_v2.SEQS not found or empty")
            except Exception as e:
                logger.debug(f"Could not load waterfall_config_v2: {e}")
            continue
        
        if choice == 's':
            seq_input = input("Enter sequence indices (space or comma-separated): ").strip()
            try:
                # Handle both space-separated and comma-separated
                seqs = [int(x.strip()) for x in seq_input.replace(',', ' ').split()]
                selected = {seq: files_by_seq[seq] for seq in seqs if seq in files_by_seq}
                if selected:
                    print(f"✓ Selected {len(selected)} sequence(s):")
                    for seq in sorted(selected.keys()):
                        print(f"  - seq_{seq:04d}  ({len(selected[seq])} files)")
                    return selected
                else:
                    print(f"✗ Could not find sequences {seqs} in available sequences")
                    print(f"Available: {seq_list}")
            except ValueError:
                print("Invalid format. Use space or comma-separated numbers")
            continue
        
        # Parse comma-separated or space-separated numbers
        try:
            indices = [int(x.strip()) - 1 for x in choice.replace(',', ' ').split()]
            selected = {}
            for idx in indices:
                if 0 <= idx < len(seq_list):
                    seq_num = seq_list[idx]
                    selected[seq_num] = files_by_seq[seq_num]
                else:
                    print(f"Invalid selection: {idx+1} (must be 1-{len(seq_list)})")
                    break
            else:
                # All indices were valid
                if selected:
                    print(f"\nSelected {len(selected)} sequence(s):")
                    for seq in sorted(selected.keys()):
                        print(f"  - seq_{seq:04d}  ({len(selected[seq])} files)")
                    return selected
        except ValueError:
            print("Invalid format. Enter comma-separated numbers, 'a', 'w', 's', or 'q'")


def init_consolidation(selected_seqs, date_str, hdf_root=None):
    """
    Initialize consolidation tracking dict for accumulating shot results into DataFrames,
    one per sequence.
    
    Parameters
    ----------
    selected_seqs : dict
        Dictionary mapping sequence number to list of HDF5 files
    date_str : str
        Date in YYYY-MM-DD format
    hdf_root : str, optional
        Root directory for reanalyzed data
    
    Returns
    -------
    dict
        Consolidation tracking dict with structure:
        {
            seq_num_1: {'path': str, 'hdf_root': str, 'rows': []},
            seq_num_2: {'path': str, 'hdf_root': str, 'rows': []},
            ...
        }
    """
    if hdf_root is None:
        hdf_root = '/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/data_reanalyzed'
    
    # Create directory structure
    year, month, day = date_str.split('-')
    data_dir = os.path.join(hdf_root, year, month, day)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create one consolidation per sequence
    consolidations = {}
    for seq_num in sorted(selected_seqs.keys()):
        seq_name = f'seq_{seq_num:04d}'
        hdf_path = os.path.join(data_dir, f'{seq_name}.hdf')
        
        logger.info(f"Consolidation for {seq_name}: {hdf_path}")
        
        consolidations[seq_num] = {
            'path': hdf_path,
            'hdf_root': hdf_root,
            'rows': []  # Accumulate rows as list of dicts
        }
    
    return consolidations


def save_consolidated_hdf(consolidations):
    """
    Save accumulated shot results to consolidated HDF files, one per sequence.
    
    Creates a MultiIndex DataFrame matching Lyse structure with:
    - Index: DatetimeIndex with 'run time' extracted from h5_attr_run time
    - Columns: MultiIndex with (parameter_name, '') structure
    
    Parameters
    ----------
    consolidations : dict
        Consolidation tracking dict with accumulated rows per sequence:
        {seq_num_1: {...}, seq_num_2: {...}, ...}
    
    Returns
    -------
    list
        List of paths to saved consolidated HDF files
    """
    try:
        saved_paths = []
        
        if not consolidations:
            logger.warning("No consolidations to save")
            return []
        
        for seq_num, consolidation in consolidations.items():
            if not consolidation or not consolidation['rows']:
                logger.warning(f"No data to consolidate for sequence {seq_num:04d}")
                continue
            
            # Create DataFrame from accumulated rows
            df_consolidated = pd.DataFrame(consolidation['rows'])
            logger.info(f"Seq {seq_num:04d}: {len(df_consolidated)} rows × {len(df_consolidated.columns)} columns")
            
            # Create DatetimeIndex from 'h5_attr_run time' column (matches Lyse structure)
            if 'h5_attr_run time' in df_consolidated.columns:
                try:
                    # Convert run time strings to datetime
                    run_times = pd.to_datetime(df_consolidated['h5_attr_run time'])
                    df_consolidated.index = run_times
                    df_consolidated.index.name = 'run time'
                    logger.info(f"  ✓ Set DatetimeIndex from 'h5_attr_run time'")
                except Exception as e:
                    logger.debug(f"  Could not create DatetimeIndex: {e}")
            
            # Separate columns for MultiIndex structure matching Lyse format:
            # ALL columns as tuples: (data_origin, column_name)
            # Globals use empty string as data_origin: ('ARPB_final_det', '')
            # Measurements use data origin: ('show_ODs_v2', 'm1_1d')
            measurement_cols = []
            global_cols = []
            
            for col in df_consolidated.columns:
                if col.startswith('globals_attr_') or col.startswith('h5_attr_'):
                    global_cols.append(col)
                else:
                    measurement_cols.append(col)
            
            # Build column data efficiently using dict before DataFrame construction
            col_data = {}
            
            # Add global columns as MultiIndex tuples with empty string as data_origin
            for col in global_cols:
                # Remove 'globals_attr_' and 'h5_attr_' prefixes to get clean column name
                if col.startswith('globals_attr_'):
                    col_name = col.replace('globals_attr_', '')
                elif col.startswith('h5_attr_'):
                    col_name = col.replace('h5_attr_', '')
                else:
                    col_name = col
                # Use ('col_name', '') format for globals (matches Lyse)
                multi_col = (col_name, '')
                col_data[multi_col] = df_consolidated[col].values
            
            # Add measurement columns with MultiIndex tuples (data_origin, col_name)
            data_origin = 'show_ODs_v2'  # This is the data source for all measurements
            for col in measurement_cols:
                multi_col = (data_origin, col)
                col_data[multi_col] = df_consolidated[col].values
            
            # Create DataFrame from dict all at once (much faster than incremental assignment)
            df_final = pd.DataFrame(col_data, index=df_consolidated.index)
            df_consolidated = df_final
            logger.info(f"  ✓ Created column structure ({len(global_cols)} globals, {len(measurement_cols)} measurements)")
            
            # Save to HDF (overwrite if exists)
            hdf_path = consolidation['path']
            try:
                # Remove existing file if present
                if os.path.exists(hdf_path):
                    logger.info(f"  Overwriting existing file: {os.path.basename(hdf_path)}")
                    os.remove(hdf_path)
                
                # Try with fixed format first (handles mixed types better)
                df_consolidated.to_hdf(hdf_path, key='data', mode='w', format='fixed')
                logger.info(f"  ✓ Saved to: {os.path.basename(hdf_path)}")
                saved_paths.append(hdf_path)
            
            except Exception as e:
                logger.debug(f"  Fixed format failed: {e}, trying pickle format")
                try:
                    # Fall back to pickle format
                    import pickle
                    pkl_path = hdf_path.replace('.hdf', '.pkl')
                    if os.path.exists(pkl_path):
                        os.remove(pkl_path)
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(df_consolidated, f)
                    logger.info(f"  ✓ Saved consolidated data (pickle format)")
                    saved_paths.append(pkl_path)
                except Exception as e2:
                    logger.error(f"  Both formats failed: {e2}")
        
        return saved_paths
    
    except Exception as e:
        logger.error(f"Error saving consolidated HDFs: {e}")
        traceback.print_exc()
        return []


# Module-level function for multiprocessing (must be picklable)
def load_consolidation_row(args):
    """Load a single file's consolidation data. Used by consolidation loading pool."""
    hdf_file, seq_mapping = args
    try:
        with h5py.File(hdf_file, 'r') as h5file:
            if 'results' in h5file and 'show_ODs_v2' in h5file['results']:
                show_ODs_group = h5file['results']['show_ODs_v2']
                
                # Extract result values
                dict_results = {}
                for key in show_ODs_group.attrs.keys():
                    dict_results[key] = show_ODs_group.attrs[key]
                for key in show_ODs_group.keys():
                    try:
                        dict_results[key] = show_ODs_group[key][()]
                    except:
                        dict_results[key] = show_ODs_group[key][...]
                
                row_data = dict(dict_results)
                row_data['shot_name'] = os.path.splitext(os.path.basename(hdf_file))[0]
                
                # Add h5 and globals attributes
                for key, value in h5file.attrs.items():
                    row_data[f"h5_attr_{key}"] = value
                if 'globals' in h5file:
                    for key, value in h5file['globals'].attrs.items():
                        row_data[f"globals_attr_{key}"] = value
                
                return (seq_mapping[hdf_file], row_data)
    except Exception as e:
        logger.debug(f"Could not load consolidation data from {os.path.basename(hdf_file)}: {e}")
    return None


def run_analysis(year, month, day, sequences=None, verbose=True):
    """
    Programmatic entry point for running show_ODs_v2 batch analysis.
    
    This function can be imported and called from other scripts to preprocess
    HDF5 files for specific dates and sequences without requiring user interaction.
    
    Parameters
    ----------
    year : int
        Year (e.g., 2026)
    month : int
        Month (1-12)
    day : int
        Day (1-31)
    sequences : list, optional
        List of sequence indices to process (e.g., [26, 30]).
        If None, will process all available sequences for that date.
    verbose : bool
        If True, print status messages. If False, suppress output (except errors).
    
    Returns
    -------
    bool
        True if analysis completed successfully, False otherwise.
    
    Examples
    --------
    >>> from run_show_ODs_batch import run_analysis
    >>> success = run_analysis(2026, 3, 24, sequences=[33, 34, 35])
    >>> if success:
    ...     print("Preprocessing complete!")
    """
    
    total_timer = Timer()
    total_timer.__enter__()
    
    # Format date string
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    
    if verbose:
        print(f"\n✓ Preprocessing for date: {date_str}")
    
    # Get HDF files for that date
    files_by_seq = get_hdf_files_for_date(date_str)
    
    if not files_by_seq:
        if verbose:
            logger.error(f"No sequences found for {date_str}")
        return False
    
    if verbose:
        logger.info(f"Found {len(files_by_seq)} sequence(s) available")
    
    # Select sequences
    if sequences is None:
        # Use all available sequences
        selected_seqs = files_by_seq
        if verbose:
            logger.info(f"No sequences specified, processing all {len(selected_seqs)} available")
    else:
        # Filter to requested sequences
        selected_seqs = {seq: files_by_seq[seq] for seq in sequences if seq in files_by_seq}
        
        # Check for missing sequences
        missing = [seq for seq in sequences if seq not in files_by_seq]
        if missing:
            if verbose:
                logger.warning(f"Requested sequences not found: {missing}")
        
        if not selected_seqs:
            if verbose:
                logger.error(f"None of the requested sequences {sequences} found for {date_str}")
            return False
        
        if verbose:
            logger.info(f"Processing {len(selected_seqs)} requested sequence(s): {sorted(selected_seqs.keys())}")
    
    # Count total files
    total_files = sum(len(files) for files in selected_seqs.values())
    if verbose:
        print(f"  Total files: {total_files}\n")
    
    # Proceed with processing (no confirmation needed when called programmatically)
    return _process_sequences(selected_seqs, date_str, total_timer, verbose)


def _process_sequences(selected_seqs, date_str, total_timer, verbose=True):
    """
    Internal function to process sequences (extracted from main() for code reuse).
    
    Parameters
    ----------
    selected_seqs : dict
        Dictionary mapping sequence numbers to file lists
    date_str : str
        Date string in format YYYY-MM-DD
    total_timer : Timer
        Timer object for tracking total execution time
    verbose : bool
        If True, print status messages
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    
    # Count total files
    total_files = sum(len(files) for files in selected_seqs.values())
    
    # Initialize consolidation DataFrames (one per sequence)
    logger.info(f"\n{'='*60}")
    logger.info(f"INITIALIZING CONSOLIDATION (ONE PER SEQUENCE)")
    logger.info(f"{'='*60}")
    init_timer = Timer()
    with init_timer:
        consolidations = init_consolidation(selected_seqs, date_str)
    logger.info(f"Consolidation initialized in {init_timer.format()}")
    
    # Process files
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING {total_files} FILES IN {len(selected_seqs)} SEQUENCE(S)")
    logger.info(f"Using {NUM_WORKERS} parallel workers")
    logger.info(f"{'='*60}")
    
    total_success = 0
    file_count = 0
    
    # Flatten file list for parallel processing
    all_files = []
    file_to_seq = {}  # Track which sequence each file belongs to
    for seq_num in sorted(selected_seqs.keys()):
        files = selected_seqs[seq_num]
        logger.info(f"\nSequence {seq_num:04d} ({len(files)} files):")
        for hdf_file in files:
            all_files.append(hdf_file)
            file_to_seq[hdf_file] = seq_num
    
    # Process in parallel (with timing)
    process_timer = Timer()
    try:
        with process_timer:
            with Pool(NUM_WORKERS) as pool:
                results = pool.imap_unordered(process_file_wrapper, [(f,) for f in all_files])
                for filename, success in results:
                    file_count += 1
                    status = "✓" if success else "✗"
                    print(f"  [{file_count}/{total_files}] {filename} ... {status}")
                    if success:
                        total_success += 1
    except Exception as e:
        logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing
        with process_timer:
            for hdf_file in all_files:
                file_count += 1
                print(f"  [{file_count}/{total_files}] {os.path.basename(hdf_file)}", end=" ... ")
                if process_file(hdf_file):
                    print("✓")
                    total_success += 1
                else:
                    print("✗")
    
    logger.info(f"File processing completed in {process_timer.format()} ({total_files/max(process_timer.elapsed, 0.001):.1f} files/sec)")
    
    # Now load all processed files and build consolidations
    logger.info(f"\nLoading analysis results into consolidations...")
    consolidation_timer = Timer()
    with consolidation_timer:
        # Parallel consolidation loading
        try:
            with Pool(NUM_WORKERS) as pool:
                # Pass tuples of (hdf_file, file_to_seq mapping)
                work_items = [(f, file_to_seq) for f in all_files]
                for result in pool.imap_unordered(load_consolidation_row, work_items):
                    if result:
                        seq_num, row_data = result
                        consolidations[seq_num]['rows'].append(row_data)
        except Exception as e:
            logger.error(f"Parallel consolidation loading failed: {e}")
            # Fall back to sequential loading (shouldn't happen, but safe)
            for hdf_file in all_files:
                result = load_consolidation_row((hdf_file, file_to_seq))
                if result:
                    seq_num, row_data = result
                    consolidations[seq_num]['rows'].append(row_data)
    
    logger.info(f"Consolidation loading completed in {consolidation_timer.format()}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_files - total_success}")
    
    # Save consolidated HDFs
    if total_success > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"SAVING CONSOLIDATED HDFS (ONE PER SEQUENCE)")
        logger.info(f"{'='*60}")
        save_timer = Timer()
        with save_timer:
            consolidated_paths = save_consolidated_hdf(consolidations)
        logger.info(f"HDF save completed in {save_timer.format()}")
        
        if consolidated_paths:
            print(f"\n✓ Consolidated data saved:")
            for path in consolidated_paths:
                print(f"  {path}")
    
    # Final timing report
    total_timer.__exit__(None, None, None)
    print(f"\n{'='*70}")
    print(f"TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"Total execution time: {total_timer.format()}")
    print(f"  - File processing: {process_timer.format()} ({total_files/max(process_timer.elapsed, 0.001):.1f} files/sec)")
    print(f"  - Consolidation loading: {consolidation_timer.format()}")
    if 'save_timer' in locals():
        print(f"  - HDF save: {save_timer.format()}")
    print(f"{'='*70}\n")
    
    return total_success == total_files


def main():
    """
    Interactive main entry point for command-line usage.
    
    This function prompts the user for date and sequence selection,
    then runs the preprocessing analysis.
    """
    
    total_timer = Timer()
    total_timer.__enter__()
    
    # Prompt for date
    date_str = prompt_for_date()
    if date_str is None:
        print("\nCancelled.")
        return False
    
    print(f"\n✓ Selected date: {date_str}")
    
    # Get HDF files for that date (all sequences)
    files_by_seq = get_hdf_files_for_date(date_str)
    
    if not files_by_seq:
        logger.error(f"No sequences found for {date_str}")
        return False
    
    logger.info(f"Found {len(files_by_seq)} sequence(s)")
    
    # Prompt for sequences
    selected_seqs = prompt_for_sequences(files_by_seq, date_str)
    if selected_seqs is None:
        print("\nCancelled.")
        return False
    
    print(f"\n✓ Will process {len(selected_seqs)} sequence(s)")
    
    # Count total files
    total_files = sum(len(files) for files in selected_seqs.values())
    print(f"  Total files: {total_files}")
    
    # Confirm before processing
    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return False
    
    # Process with interactive prompts (verbose=True for command-line usage)
    return _process_sequences(selected_seqs, date_str, total_timer, verbose=True)


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")

        traceback.print_exc()
        sys.exit(1)
