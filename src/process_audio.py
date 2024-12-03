# Command line interface to process audio data.
# Run `python src/process_audio.py -h` to display help.

from multiprocessing import freeze_support

import argparse
from misc import log
import sys
import pandas as pd
import os
import shutil
import time

# Add BirdNET-Analyzer to the Python path
# print('process_audio.py')
birdnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'submodules', 'BirdNET-Analyzer'))
if birdnet_path not in sys.path:
    print(f'Adding BirdNET-Analyzer to sys.path {birdnet_path}')
    print('Please wait...')
    sys.path.append(birdnet_path)
from analyze import *
import config as cfg
import utils
from segments import *

def get_relative_paths(root, extension):
    return [os.path.relpath(os.path.join(dp, f), root)
            for dp, dn, filenames in os.walk(root) for f in filenames if f.endswith(f'.{extension}')]

def get_rel_dirs_with_files(root_dir, extension):
    extension = f'.{extension}'
    dirs = set()
    for dir, _, filenames in os.walk(root_dir):
        if any(file.endswith(extension) for file in filenames):
            dirs.add(os.path.relpath(dir, root_dir))
    return dirs

# Wrapper for audio.process_audio process_file_or_dir
def process(
        in_path, # path to a file or directory
        out_dir_path,
        rtype = 'csv',
        target_model_filepath = None,
        slist = None,
        use_ensemble = False,
        ensemble_weights = None,
        min_confidence = 0.0,
        threads        = 8,
        cleanup        = True
):
    if in_path == '' or out_dir_path == '':
        log.print_error('Input and output paths must be specified to calculate predictions')
        return

    if use_ensemble and (target_model_filepath == None):
        log.print_error('Target model path required for ensemble')
        return
    
    out_dir_path = os.path.join(out_dir_path, 'predictions')

    if target_model_filepath == 'None':
        target_model_filepath = None

    # Reset config defaults for repeat processes
    cfg.LABELS_FILE = "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
    cfg.CUSTOM_CLASSIFIER = None

    models = []
    model_tags = []

    if (target_model_filepath == None) or use_ensemble:
        models.append(None)
        model_tags.append('source')
        model = 'source'

    if (target_model_filepath != None) or use_ensemble:
        models.append(target_model_filepath)
        model_tags.append('target')

    # For each requested model, analyze all files
    for i, model in enumerate(models):
        model_tag = model_tags[i]
        print(f"Processing predictions for {in_path} with {model_tag} model...")

        time_start = time.time()

        if use_ensemble:
            working_out_dir_path = os.path.join(out_dir_path, 'temp', model_tag)
        else:
            working_out_dir_path = out_dir_path

        args = argparse.Namespace(
            i=in_path,
            o=working_out_dir_path,
            lat=-1, lon=-1, week=-1,
            slist=slist,
            sensitivity=1.0,
            min_conf=min_confidence,
            overlap=0.0,
            rtype=rtype,
            output_file=None,
            threads=threads,
            batchsize=1,
            locale='en',
            sf_thresh=0.03,
            classifier=model, # None for source
            fmin=cfg.SIG_FMIN, fmax=cfg.SIG_FMAX,
            skip_existing_results=True
        )
        analyze_main_wrapper(args, birdnet_path)
    
    if use_ensemble:
        print('Calculating ensemble predictions...')
        source_dir = os.path.join(out_dir_path, 'temp', 'source')
        target_dir = os.path.join(out_dir_path, 'temp', 'target')

        weights = pd.read_csv(ensemble_weights, sep='\t')
        weights['Common Name'] = weights['Label'].str.split('_').str[-1]

        if args.rtype == 'table':
            extension = 'txt'
        elif args.rtype == 'csv':
            extension = 'csv'

        for rel_path in get_relative_paths(source_dir, extension):
            source_file = os.path.join(source_dir, rel_path)
            target_file = os.path.join(target_dir, rel_path)
            result_file = os.path.join(out_dir_path, rel_path)
            if os.path.exists(target_file):

                if args.rtype == 'table':
                    predictions_source = pd.read_csv(source_file, sep='\t')
                    predictions_target = pd.read_csv(target_file, sep='\t')
                    predictions_source = predictions_source.drop(columns=['Selection'])
                    predictions_target = predictions_target.drop(columns=['Selection'])
                elif args.rtype == 'csv': 
                    predictions_source = pd.read_csv(source_file)
                    predictions_target = pd.read_csv(target_file)
                    predictions_source = predictions_source.rename(columns={'Common name': 'Common Name'})
                    predictions_target = predictions_target.rename(columns={'Common name': 'Common Name'})
                else:
                    log.print_error(f'Unsupported rtype {args.rtype}')
                    return
                
                # Merge the predictions
                shared_cols = list(predictions_source.columns)
                shared_cols.remove('Confidence')
                predictions_ensemble = pd.merge(predictions_source, predictions_target, on=shared_cols, how='outer', suffixes=('_source', '_target'))

                # Calculate ensemble confidence
                predictions_ensemble = pd.merge(predictions_ensemble, weights, on='Common Name', how='left')
                predictions_ensemble['Confidence'] = (
                    predictions_ensemble['Confidence_source'].fillna(0.0) * predictions_ensemble['Weight_Source'] +
                    predictions_ensemble['Confidence_target'].fillna(0.0) * predictions_ensemble['Weight_Target']
                )
                predictions_ensemble['Confidence'] = predictions_ensemble['Confidence'].fillna(0.0)
                predictions_ensemble = predictions_ensemble.drop(columns=['Weight_Source', 'Weight_Target'])

                # Write result to file
                predictions_ensemble = predictions_ensemble.sort_index(axis=1)
                if not os.path.exists(os.path.dirname(result_file)):
                    os.makedirs(os.path.dirname(result_file))
                if args.rtype == 'table':
                    predictions_ensemble.insert(0, 'Selection', predictions_ensemble.index + 1)
                    predictions_ensemble.to_csv(result_file, sep='\t', index=False)
                elif args.rtype == 'csv':
                    cols_leading = ['Start (s)', 'End (s)']
                    col_order = cols_leading + [col for col in predictions_ensemble.columns if col not in cols_leading]
                    predictions_ensemble = predictions_ensemble[col_order]
                    predictions_ensemble = predictions_ensemble.rename(columns={'Common Name': 'Common name'})
                    predictions_ensemble.to_csv(result_file, index=False)
            else:
                log.print_warning(f"Matching file not found for {rel_path} in target directory.")
        if cleanup:
            shutil.rmtree(os.path.join(out_dir_path, 'temp'))

    time_end = time.time()
    time_elapsed = (time_end - time_start)/60.0

    log.print_success(f'Finished processing predictions ({time_elapsed:.1f} min)')

    return

def segment(
    in_audio_path,
    in_predictions_path,
    extension,
    out_dir_path,
    min_conf,
    max_segments,
    seg_length,
    threads
):
    if (in_audio_path == '') or (in_predictions_path == '') or (out_dir_path == ''):
        log.print_error('Input audio, predictions, and output paths must be specified to extract audio segments')
        return

    time_start = time.time()

    rel_dirs = get_rel_dirs_with_files(in_predictions_path, extension)
    for dir in rel_dirs:
        print(f'Extracting audio segments from {dir}')

        args = argparse.Namespace(
            audio=in_audio_path,
            results=os.path.join(in_predictions_path, dir),
            o=os.path.join(out_dir_path, dir),
            min_conf=min_conf,
            max_segments=max_segments,
            seg_length=seg_length,
            threads=threads
        )
        segments_main_wrapper(args)

    time_end = time.time()
    time_elapsed = (time_end - time_start)/60.0

    log.print_success(f'Finished extracting audio segments ({time_elapsed:.1f} min)')


def main(args):
    
    process(
        in_path                         = args.in_path,
        out_dir_path                    = args.out_path_predictions,
        rtype                           = args.out_filetype,
        target_model_filepath           = args.target_model_filepath,
        slist                           = args.slist, 
        use_ensemble                    = args.use_ensemble,
        ensemble_weights                = args.ensemble_weights,
        min_confidence                  = args.min_confidence,
        threads                         = args.threads,
        cleanup                         = True
    )

    extension = args.out_filetype
    if extension == "table":
        extension = "txt"

    segment(
        in_audio_path = args.in_path,
        in_predictions_path = os.path.join(args.out_path_predictions, 'predictions'),
        extension = extension,
        out_dir_path = os.path.join(args.out_path_predictions, 'segments'),
        min_conf = args.min_confidence,
        max_segments = 28800, # number of 3-second segments in 24 hours
        seg_length = 3.0,
        threads = args.threads
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with required and optional arguments")

    # Required arguments
    parser.add_argument("in_path",      type=str, help="Absolute path to a single audio file or a directory containing audio files (will search the directory tree recursively)")
    parser.add_argument("out_path_predictions", type=str, help="Absolute path to the output directory")

    # Optional arguments
    parser.add_argument("--out_filetype", type=str, help="Supported file types: 'csv' or 'table' (Raven selection table)")
    parser.add_argument("--target_model_filepath", type=str, help="Relative path to target model .tflite file")
    parser.add_argument("--slist", type=str, help="Relative path to ensemble labels .txt file")
    parser.add_argument("--use_ensemble", action="store_true", help="Flag to use source and target models together as an ensemble")
    parser.add_argument("--ensemble_weights", type=str, help="Model-specific class weights for ensemble")
    parser.add_argument("--min_confidence", type=float, help="Minimum confidence score to retain a detection (float)")
    parser.add_argument("--threads", type=int, help="Number of cores used by the processing pool (<= number of physical cores available on your computer) (int)")

    # TODO: these hard-coded defaults are here for debugging; remove for distribution
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        log.print_warning('No arguments provided. Using default values...')
        # Define default values
        args = parser.parse_args([
            "/Users/giojacuzzi/Desktop/input",
            "/Users/giojacuzzi/Desktop/output",
            "--out_filetype", "csv",
            "--slist", "models/ensemble/ensemble_species_list.txt",
            "--target_model_filepath",  "models/target/OESF_1.0/OESF_1.0.tflite",
            "--use_ensemble",
            "--ensemble_weights", "models/ensemble/ensemble_weights.txt",
            "--min_confidence", "0.5",
            "--threads", "8"
            # "--cleanup"
        ])
    main(args)
