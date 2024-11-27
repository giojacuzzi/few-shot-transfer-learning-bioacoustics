from utils.log import *
from utils.files import *
from utils.bnl import *
from birdnetlib.analyzer import Analyzer, MODEL_VERSION
import birdnetlib.analyzer
from birdnetlib import Recording 
import os
import pandas as pd
from subprocess import *
import shutil
from multiprocessing import Pool
import time
from itertools import repeat
from pathlib import Path
from . import sound_separation

def find_dirs_containing_filetype(root_path, filetype):
    dirs = []

    if any(f.suffix == filetype for f in root_path.iterdir() if f.is_file()):
        dirs.append(root_path)

    for d in root_path.rglob('*'): # recursive
        if d.is_dir():
            if any(file.suffix == filetype for file in d.iterdir()):
                dirs.append(d)
    return sorted(dirs)

def find_files_in_dir(dir_path, filetype):
    files = [f for f in dir_path.glob(f'*{filetype}')]
    return sorted(files)

# Process the given file(s) with analyzer(s) and save the resulting predictions to file(s)
def process_file_or_dir(
        in_path, # path to a file or directory
        in_filetype = None,
        out_dir_path        = '', # path to an output directory
        retain_dir_tree       = True, # directory structure below root_dir will be retrained in out_dir_path
        target_model_filepath = None,
        source_labels_filepath = 'src/classification/species_list/species_list_OESF.txt',
        target_labels_filepath = None,
        use_ensemble = False,
        ensemble_class_model_selections = None,
        min_confidence = 0.0,
        retain_logit_score  = False,
        n_processes    = 8, # cores per batch
        num_separation = 1,
        cleanup        = True,
        sort_by        = '',
        ascending      = True,
        out_filetype   = '',
        digits = 3,
):
    start_time_process = time.time()
    # print('process_file_or_dir')

    if (out_dir_path != '' and out_filetype == '') or (out_dir_path == '' and out_filetype != ''):
        print_error(f'Must specify an output path and file type together')
        return
    out_dir_path = Path(out_dir_path)

    if (source_labels_filepath != '' and source_labels_filepath != None):
        source_labels_filepath = Path(source_labels_filepath)
    if (target_model_filepath != '' and target_model_filepath != None):
        target_model_filepath = Path(target_model_filepath)
    if (target_labels_filepath != '' and target_labels_filepath != None):
        target_labels_filepath = Path(target_labels_filepath)
    if (ensemble_class_model_selections != '' and ensemble_class_model_selections != None):
        ensemble_class_model_selections = Path(ensemble_class_model_selections)

    in_path = Path(in_path)
    if in_path.is_file(): # -------------------------------------------
        print(f'Processing file {in_path}')
        result = process_file(
            in_filepath    = in_path,
            out_dir_path   = out_dir_path,
            custom_model_filepath = target_model_filepath,
            source_labels_filepath = source_labels_filepath,
            target_labels_filepath = target_labels_filepath,
            use_ensemble   = use_ensemble,
            ensemble_class_model_selections = ensemble_class_model_selections,
            min_confidence = min_confidence,
            retain_logit_score  = retain_logit_score,
            num_separation = num_separation,
            cleanup        = cleanup,
            sort_by        = sort_by,
            ascending      = ascending,
            out_filetype   = out_filetype,
            digits = digits
        )
        return(result)

    elif in_path.is_dir(): # -------------------------------------------
        
        dirs = find_dirs_containing_filetype(in_path, in_filetype)
        for dir in dirs:
            print(f'Processing directory {dir}')

            start_time_dir = time.time()
            in_filepaths = find_files_in_dir(dir, in_filetype)
            n_files = len(in_filepaths)
            
            if n_files == 0:
                print_warning(f'No {in_filetype} files found in {dir}. Skipping...')
                continue

            processes_available = min(n_files, n_processes)
            print(f'Executing {processes_available} processes for {n_files} files...')

            out_dir_paths = []
            if retain_dir_tree:
                for f in in_filepaths:
                    f = Path(f)
                    if (f.parent != in_path):
                        relative_path = f.parent.relative_to(in_path)
                        out_dir_paths.append(out_dir_path/relative_path)
                    else:
                        out_dir_paths = [out_dir_path] * processes_available
            else:
                out_dir_paths = [out_dir_path] * processes_available

            with Pool(processes_available) as pool: # start process pool for all files in directory
                pool.starmap(process_file, zip(
                    in_filepaths, # in_filepath
                    out_dir_paths, # out_dir_path
                    repeat(target_model_filepath), # custom_model_filepath
                    repeat(source_labels_filepath), # source_labels_filepath
                    repeat(target_labels_filepath), # target_labels_filepath
                    repeat(use_ensemble), # use_ensemble
                    repeat(ensemble_class_model_selections), # ensemble_class_model_selections
                    repeat(min_confidence), # min_confidence
                    repeat(retain_logit_score), # retain_logit_score
                    repeat(num_separation), # num_separation
                    repeat(cleanup), # cleanup
                    repeat(sort_by), # sort_by
                    repeat(ascending), # ascending
                    repeat(out_filetype), # out_filetype
                    repeat(digits) # digits
                ))


            end_time_dir = time.time()
            print_success(f'Finished directory {dir}\n({round((end_time_dir - start_time_dir)/60.0, 2)} min)')

    else: # -------------------------------------------
        print_error(f'Failed to find input path {in_path}')
        return
    
    end_time_process = time.time()
    print_success(f'Finished processing\n({round((end_time_process - start_time_process)/3600.0, 2)} hr)')

# Run a BirdNET analyzer model with or without source separation.
# path - path to a .wav file
# cleanup - remove any temporary files created during analysis
# num_separation - the number of sound channels to separate. '1' will leave the original file unaltered.
def run_analyzer(filepath, analyzer, min_confidence, retain_logit_score=False, num_separation=1, cleanup=True):
    
    if (num_separation > 1):
        files = sound_separation.separate(filepath, num_separation)
    else:
        files = [filepath] # original file, no sound separation

    detections = pd.DataFrame()
    for file in files: # Obtain detections across file(s)

        birdnetlib.analyzer.APPLY_SIGMOID = (not retain_logit_score) # Overwrite flag to return logit values instead of sigmoid transforms
        
        recording = Recording(analyzer=analyzer, path=file, min_conf=min_confidence)
        recording.minimum_confidence = min_confidence # necessary override to enforce 0.0 case
        recording.analyze()
        file_detections = pd.DataFrame(recording.detections)
        print(f'analyze_detections found {str(len(file_detections))} detections')

        file_detections['file_origin'] = os.path.basename(file)
        if (len(file_detections) > 0):
            detections = pd.concat([detections, file_detections], ignore_index=True)
        else:
            print('no detections')
    
    if num_separation == 1:
        return detections
    else:
        if cleanup:  # Aggregate detections for source separation
            shutil.rmtree(os.path.dirname(files[0]))

        # Find indices of maximum confidence values for each unique start_time and common_name combination
        aggregated_detections = pd.DataFrame(detections).sort_values('start_time')
        i = aggregated_detections.groupby(['start_time', 'common_name'])['confidence'].idxmax()
        aggregated_detections = aggregated_detections.loc[i]
        return aggregated_detections

# Run the analyzer on the given file and save the resulting detections to file
def process_file(
        in_filepath,
        out_dir_path,
        custom_model_filepath,
        source_labels_filepath,
        target_labels_filepath,
        use_ensemble,
        ensemble_class_model_selections,
        min_confidence,
        retain_logit_score,
        num_separation,
        cleanup,
        sort_by,
        ascending,
        out_filetype,
        digits
):
    if use_ensemble and (custom_model_filepath is None or ensemble_class_model_selections is None or target_labels_filepath is None):
        print_error('Must provide target model and labels filepaths and class model selections for a model ensemble')
        return()

    print(f'Processing file {in_filepath}...')
    out_dir_path = Path(out_dir_path)

    file_out = os.path.basename(os.path.splitext(in_filepath)[0]) + out_filetype
    path_out = out_dir_path / file_out

    analyzers = []
    analyzer_labels = {}
    if custom_model_filepath is None or use_ensemble: # Use default pre-trained source model
        source_labels_filepath = os.path.abspath(source_labels_filepath)
        print(f'Initializing pre-trained analyzer with species list {source_labels_filepath}')
        analyzer = Analyzer(custom_species_list_path=source_labels_filepath)
        analyzers.append(analyzer)
        analyzer_labels[analyzer] = 'source'
        # TODO: cache analyzer
    
    if custom_model_filepath is not None: # Use custom target model
        custom_model_filepath = os.path.abspath(custom_model_filepath)
        target_labels_filepath = os.path.abspath(target_labels_filepath)
        print(f'Initializing custom analyzer {custom_model_filepath} with species list {target_labels_filepath}')
        analyzer = Analyzer(classifier_model_path=custom_model_filepath, classifier_labels_path=target_labels_filepath)
        analyzers.append(analyzer)
        analyzer_labels[analyzer] = 'target'
        # TODO: cache analyzer

    already_analyzed = list_base_files_by_extension(out_dir_path, out_filetype)
    already_analyzed = [f.rstrip(out_filetype) for f in already_analyzed]
    if out_dir_path != '' and ((os.path.splitext(os.path.basename(in_filepath))[0]) in already_analyzed):
        print_warning(f'{os.path.basename(in_filepath)} already analyzed. Skipping...')
        return

    # Run analyzer model(s) to obtain detections
    start_time_file = time.time()
    predictions = pd.DataFrame()
    for analyzer in analyzers:
        print(f'process_file {analyzer_labels[analyzer]}')

        try:
            result = run_analyzer(
                filepath = in_filepath,
                analyzer = analyzer,
                min_confidence = min_confidence,
                retain_logit_score = retain_logit_score,
                num_separation = num_separation,
                cleanup = cleanup
            )
        except Exception as e:
            print_error(f'Unable to process file {in_filepath}: {e}')
            if out_filetype != '':
                if not os.path.exists(os.path.dirname(path_out)):
                    os.makedirs(os.path.dirname(path_out))
                p = Path(path_out)
                np = p.with_name("ERROR_" + p.name)
                with open(np, "w") as file:
                    file.write("ERROR")
            return None

        # Only keep essential data
        if retain_logit_score:
            col_names = ['common_name','confidence','logit','start_time']
        else:
            col_names = ['common_name','confidence','start_time']

        if not result.empty:
            # If analyzer returned raw logit values in the 'confidence' column, create
            # a unique column each for logit values and sigmoid confidence scores.
            if retain_logit_score:
                result = result.rename(columns={'confidence': 'logit'})
                result['confidence'] = sigmoid_BirdNET(result['logit'])

            result = result[col_names]

        else:
            result = pd.DataFrame(columns=col_names)
        
        result['model'] = analyzer_labels[analyzer]
        predictions = pd.concat([predictions, result], axis=0, ignore_index=True)

    if use_ensemble:
        # Create an column 'confidence' for ensemble predictions, retaining prediction value from either source or target model according to the class model selections file
        pred_source = predictions[predictions['model'] == 'source'].drop(columns='model')
        pred_target = predictions[predictions['model'] == 'target'].drop(columns='model')
        predictions = pd.merge(pred_source, pred_target, on=['common_name', 'start_time'], how='outer', suffixes=('_source', '_target'))
        ensemble_class_model_selections = os.path.abspath(ensemble_class_model_selections)
        class_model_selections = pd.read_csv(ensemble_class_model_selections)
        predictions['common_name_lower'] = predictions['common_name'].str.lower()
        merged_df = predictions.merge(class_model_selections, left_on='common_name_lower', right_on='label', how='left')
        merged_df['confidence'] = merged_df.apply(
            lambda row: row['confidence_target'] if row['model_selection'] == 'target' else row['confidence_source'], axis=1
        )
        predictions['confidence'] = merged_df['confidence']
        predictions = predictions.drop(columns='common_name_lower')

    # Cleanup and sorting
    predictions = predictions.drop(columns='model', errors='ignore')
    predictions = predictions.round(decimals=digits)
    cols = [col for col in predictions.columns if col != 'common_name']
    cols.sort()
    predictions = predictions[['common_name'] + cols]
    if sort_by != '':
        predictions = predictions.sort_values(sort_by, ascending=ascending)

    # Save results to file
    if out_dir_path != '' and out_filetype != '':
        if not os.path.exists(os.path.dirname(path_out)):
            os.makedirs(os.path.dirname(path_out))
        
        if out_filetype == '.csv':
            pd.DataFrame.to_csv(predictions, path_out, index=False)
        elif out_filetype == '.parquet':
            pd.DataFrame.to_parquet(predictions, path_out, index=False)
        else:
            print_error(f'Incompatible output filetype {out_filetype}')

    end_time_file = time.time()
    time_delta_sec = (end_time_file - start_time_file)
    if time_delta_sec < 60:
        print_success(f'Finished file {in_filepath}\n({round(time_delta_sec,1)} sec)')
    else:
        print_success(f'Finished file {in_filepath}\n({round(time_delta_sec/60.0, 2)} min)')
    return predictions
