import os
from .log import *

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
