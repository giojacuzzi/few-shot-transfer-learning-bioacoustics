import os

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
