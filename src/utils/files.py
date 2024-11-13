from .log import *
import os
import re
import pandas as pd
import numpy as np
from utils.log import *
import sys

# File with table containing ID and ecological data for monitored species
species_list_filepath = 'data/species/species_list - complete.csv'

#
def find_file_full_path(top_directory, filename):
    for root, dirs, files in os.walk(top_directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Function to parse species name, confidence, serial number, date, and timecfrom annotation filename,
# e.g. "Yellow-rumped Warbler-0.4231688380241394_SMA00399_20200628_204241.wav"
def parse_metadata_from_annotation_file(filename):

    # print(filename)

    # Regular expression pattern to match the filename
    pattern = r'^(.+)-([\d.]+)_(\w+)_(\d{8})_(\d{6})(?:_([\d.]+))?\.(\w+)$'
    match = re.match(pattern, filename)

    if match:
        species    = match.group(1)
        confidence = float(match.group(2))
        serialno   = match.group(3)
        date       = match.group(4)
        time       = match.group(5)
        return species, confidence, serialno, date, time
    else:
        print_warning("Unable to parse info from filename:", filename)

# Function to parse serial number, date, and time from a raw detection audio filename,
# e.g. "SMA00556_20200526_050022_3400.8122" or "_SMA00309_20200424_031413"
def parse_metadata_from_detection_audio_filename(filename):
    # print(f"parse_metadata_from_detection_audio_filename {filename}")
    pattern = r'SMA(\d+)_([0-9]{8})_([0-9]{6})$'
    match = re.search(pattern, filename)
    if match:
        serial_no = 'SMA' + match.group(1)
        date = match.group(2)
        time = match.group(3)
        return serial_no, date, time
    else:
        print_error(f'Unable to parse info from filename: {filename}')
        return None
 

# Function to parse species name, confidence, serial number, date, and time from raw audio filepath,
# e.g. ".../Deployment6/SMA00399_20200619/SMA00399_20200619_040001.wav"
def parse_metadata_from_raw_audio_filepath(filepath):

    # Regular expression pattern to match the deployment number
    deployment = re.findall(".Deployment([\d.]+)", filepath)
    if (len(deployment) != 1):
        print_warning(f'Could not parse Deployment token from {filepath}')
        deployment = np.nan
    else:
        deployment = deployment[0]

    # Regular expression pattern to match the filename
    pattern = r'(?P<serialno>\S+?)_(?P<date>\d{8}?)_(?P<time>\d{6}?)\.(?P<type>\w+?)$'
    match = re.search(pattern, os.path.basename(filepath))
    if match:
        serialno = match.group('serialno')
        date     = match.group('date')
        time     = match.group('time')
        return deployment, serialno, date, time
    else:
        print_error(f'Unable to parse info from filepath: {filepath}')

# Find all selection table files under a root directory
def find_files(directory, suffix=None, prefix=None, exclude_dirs=[]):

    results = []
    for root, dirs, files in os.walk(directory):
        if any(exclude_dir in root for exclude_dir in exclude_dirs):
            continue
        for file in files:
            suffix_match = (suffix is None) or (suffix is not None and file.endswith(suffix))
            prefix_match = (prefix is None) or (prefix is not None and file.startswith(prefix))
            if suffix_match and prefix_match:
                results.append(os.path.join(root, file))
    return results

# Scrape serial number, date, and time from Song Meter filename
def parse_metadata_from_filename(path):
    filename = os.path.basename(path)
    substrs = filename.split('.')[0].split('_')
    if len(substrs) != 3:
        # print_warning('Could not get metadata from filename')
        return
    date = substrs[1]
    time = substrs[2]
    return({
        'serial_no': substrs[0],
        'date':      date,
        'year':      date[0:4],
        'month':     date[4:6],
        'day':       date[6:8],
        'time':      time,
        'hour':      time[0:2],
        'min':       time[2:4],
        'sec':       time[4:6],
    })

def getDirectoriesWithFiles(path, filetype):
    directoryList = []
    # print(f'YO: {os.listdir(path)}')
    # print(len(os.listdir(path)))
    if os.path.isfile(path):
        print('returning from isfile!')
        return []
    # print(f'found {len([f for f in os.listdir(path)])} files')
    # Add dir to directorylist if it contains files of filetype
    files_in_dir = [f for f in os.listdir(path)]
    if len(files_in_dir) > 0:
        files_of_type = [f for f in files_in_dir if f.endswith('.wav')]
        # print(f'endswith {len(files_of_type)}')
        # print(f'adding path {path}')
        directoryList.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            # print(f'adding a dir {d}')
            directoryList += getDirectoriesWithFiles(new_path, filetype)
    return directoryList

def list_base_files_by_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(extension):
                file_list.append(filename)
    return file_list

def list_files_in_directory(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# Returns a dataframe of raw metadata from all raw acoustic monitoring data under the given directories
def get_raw_metadata(dirs=[], overwrite=False):

    metadata_filepath = 'data/metadata/raw_metadata.csv'

    # Return previously processed annotations unless overwrite requested
    if os.path.exists(metadata_filepath) and not overwrite:
        print(f'Retrieving metadata from {metadata_filepath}')
        return pd.read_csv(metadata_filepath, index_col=None)
    
    if (overwrite and len(dirs) == 0) or (not overwrite and not os.path.exists(metadata_filepath) and len(dirs) == 0):
        print_error('At least one directory must be provided to write metadata to file')
        return
    
    print('Retrieving metadata from raw data...')

    # Declare 'evaluated_detections' - Get the list of files (detections) that were actually evaluated by annotators
    # These are all the .wav files under each site-date folder in e.g. 'annotation/data/_annotator/2020/Deployment4/SMA00410_20200523'
    filepaths = []
    for dir in dirs:
        filepaths.extend(find_files(dir, suffix='.wav'))

    if len(filepaths) == 0:
        print_error('No .wav files found')
        exit()
    else:
        print(f'Found {len(filepaths)} .wav files')

    # Parse metadata from the .wav filenames
    file_metadata = list(map(lambda f: parse_metadata_from_raw_audio_filepath(f), filepaths))
    deployments, serialnos, dates, times = zip(*file_metadata)

    file_metadata = pd.DataFrame({
        'filepath':   filepaths,
        'deployment': deployments,
        'date':       dates,
        'time':       times,
        'serialno':   serialnos
    })
    file_metadata = file_metadata.sort_values(by=['deployment','serialno','date', 'time'])

    file_metadata.to_csv(metadata_filepath, index=False)
    print_success(f'Saved annotations to {metadata_filepath}')
    return file_metadata

# Returns a dataframe of site deployment metadata from a .csv downloaded from google drive
def get_site_deployment_metadata(year):
    if year != 2020:
        print_error('Only 2020 currently supported')
    site_deployment_metadata_path = 'data/metadata/OESF Deployments (Total) - 2020.csv'
    site_deployment_metadata = pd.read_csv(site_deployment_metadata_path, parse_dates=['SurveyDate'], date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y'))
    return site_deployment_metadata

# Load a Raven selection table (tab-separated .txt) as a pandas dataframe and validate contents
def load_raven_selection_table(table_file, cols_needed = ['species', 'begin file', 'file offset (s)', 'delta time (s)'], rename_species_to_label = False):
    table = pd.read_csv(table_file, sep='\t') # Load the file as a dataframe

    if table.empty:
        print_warning(f'{os.path.basename(table_file)} has no selections.')

    # Clean up data by normalizing column names and species values to lowercase
    table.columns = table.columns.str.lower()
    if 'species' in table.columns:
        table['species'] = table['species'].astype(str).str.lower()
    if 'class' in table.columns:
        table['class'] = table['class'].astype(str).str.lower()
    
    if rename_species_to_label:
        table.rename(columns=lambda x: 'label' if x.lower() == 'species' else x, inplace=True)

    # Check the validity of the table
    if not all(col in table.columns for col in cols_needed):
        missing_columns = [col for col in cols_needed if col not in table.columns]
        print_warning(f"Missing columns {missing_columns} in {table_file}.")
        return table
    
    # if table.isna().any().any():
    #     print_warning(f'{os.path.basename(table_file)} contains NaN values')
    
    table = table[cols_needed] # Discard unnecessary data
    return table
