# Evaluate audio sample level performance of a custom target model on an evaluation dataset (e.g. validation, test), and compare performance with a pre-trained source model.
#
# Input:
# - Name stub of target model to evaluate from directory "models/target" (e.g. "OESF_1.0")
# - Path to the root directory containing all audio files for evaluation
# - Table containing evaluation dataset annotations (e.g. "data/test/test_data_annotations.csv")
# - Source and target labels lists (e.g. "models/source/source_species_list.txt" and "models/target/OESF_1.0/OESF_1.0_Labels.txt")
#
# Output:
# - Intermediate prediction scores for source and target models at "data/interm/{target_model_stub}/{model}"
# - Threshold performance values and final performance metrics results at "results/{target_model_stub}/sample_perf"
#
# Afterwards, plot results with figs/fig_sample_perf.R

# CHANGE ME ##############################################################################
evaluation_dataset = 'test' # 'validation' or 'test'
target_model_stub  = 'OESF_1.0' # Name of the target model to evaluate from directory "models/target/{target_model_stub}""; e.g. 'custom_S1_N100_LR0.001_BS10_HU0_LSFalse_US0_I0' or None to only evaluate pre-trained model
evaluation_audio_dir_path = '/Users/giojacuzzi/Library/CloudStorage/GoogleDrive-giojacuzzi@gmail.com/My Drive/Research/Projects/OESF/transfer learning/data/test' # Path to root directory containing all audio files for evaluation
overwrite = False
plot_precision_recall = False
##########################################################################################

from misc.log import *
from misc import files
from perf.perf_metrics import *
import numpy as np
import os
import pandas as pd
import shutil
import sys

if not os.path.exists(f'results/{target_model_stub}/sample_perf'):
    os.makedirs(f'results/{target_model_stub}/sample_perf')

if evaluation_dataset != 'validation' and evaluation_dataset != 'test':
    print_error('Invalid evaluation dataset')
    sys.exit()
if target_model_stub == None and evaluation_dataset == 'validation':
    print_error('Evaluating with a validation dataset requires a custom model specified')
    sys.exit()

# Load class labels
source_class_labels = pd.read_csv(os.path.abspath(f'models/source/source_species_list.txt'), header=None)[0].tolist()
target_class_labels = pd.read_csv(os.path.abspath(f'models/target/target_species_list.txt'), header=None)[0].tolist()

novel_labels_to_evaluate = [l for l in target_class_labels if l not in source_class_labels]
preexisting_labels_to_evaluate = source_class_labels #[l for l in target_class_labels if l in source_class_labels]
target_labels_to_evaluate = target_class_labels
all_labels = list(set(source_class_labels + target_class_labels))

print(f"{len(preexisting_labels_to_evaluate)} preexisting labels to evaluate:")
print(preexisting_labels_to_evaluate)
print(f"{len(target_labels_to_evaluate)} target labels to evaluate:")
print(target_labels_to_evaluate)

# Output config
sort_by      = 'confidence' # Column to sort dataframe by
ascending    = False        # Column sort direction
save_to_file = True         # Save output to a file
if evaluation_dataset == 'validation' and target_model_stub != None: # Validation dataset with target model
    out_dir = f'data/interim/validation/custom/{target_model_stub}'
    target_model_dir_path = f'models/target/{target_model_stub}'
elif evaluation_dataset == 'test': # Test dataset
    if target_model_stub == None: # Source model
        out_dir = 'data/interim/source'
    else: # Target model
        out_dir = f'data/interim/{target_model_stub}/sample_perf' #f'data/test/{target_model_stub}'
    target_model_dir_path = f'models/target/{target_model_stub}'

# Analyzer config
min_confidence = 0.0   # Minimum confidence score to retain a detection (only used if apply_sigmoid is True)
cleanup        = True  # Keep or remove any temporary files created through analysis
n_processes = 1

source_analyzer_filepath = None # 'None' will default to the source model
source_labels_filepath   = 'models/source/source_species_list.txt'
target_analyzer_filepath = f'{target_model_dir_path}/{target_model_stub}.tflite'
target_labels_filepath   = f'{target_model_dir_path}/{target_model_stub}_Labels.txt'
training_data_path       = 'data/training'

def remove_extension(f):
    return os.path.splitext(f)[0]

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Get the evaluation data as a dataframe with columns:
    # 'path' - full path to the audio file
    # 'file' - the basename of the audio file

    # TODO
    if evaluation_dataset == 'validation':
        development_data = pd.read_csv(f'{target_model_dir_path}/combined_files.csv')
        evaluation_data = development_data[development_data['dataset'] == 'validation']
        evaluation_data.loc[:, 'file'] = evaluation_data['file'].apply(remove_extension)

    elif evaluation_dataset == 'test':
        evaluation_data = pd.read_csv('data/test/test_data_annotations.csv') # see pretest_consolidate_test_annotations.py
        evaluation_data = evaluation_data[evaluation_data['target'].isin(all_labels)]
        evaluation_data['labels'] = evaluation_data['labels'].fillna('')

    out_dir_source = out_dir + '/source'
    out_dir_target = out_dir + '/target'
    if target_model_stub == None:
        models = [out_dir_source]
    else:
        models = [out_dir_source, out_dir_target]

    # Normalize file paths to support both mac and windows
    out_dir = os.path.normpath(out_dir)

    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    if os.path.exists(out_dir):
        print_warning('Raw audio data already processed, loading previous detection results for model(s)...')
    else:
        import process_audio

        # Process evaluation examples with both classifiers -----------------------------------------------------------------------------
        if target_model_stub != None:
            # Custom target classifier model
            print(f"Processing evaluation set with target model {target_analyzer_filepath}...")
            process_audio.process(
                in_path = evaluation_audio_dir_path,
                out_dir_path = out_dir_target,
                rtype = 'csv',
                target_model_filepath = target_analyzer_filepath,
                slist = target_labels_filepath,
                min_confidence = min_confidence,
                threads = n_processes,
                cleanup = cleanup
            )
        # Pre-trained source classifier model
        print(f"Processing evaluation set with source model...")
        process_audio.process(
            in_path = evaluation_audio_dir_path,
            out_dir_path = out_dir_source,
            rtype = 'csv',
            slist = source_labels_filepath,
            min_confidence = min_confidence,
            threads = n_processes,
            cleanup = cleanup
        )
    
    performance_metrics = pd.DataFrame()

    for model in models:
        print(f'BEGIN MODEL EVALUATION {model} (sample level) ---------------------------------------------------------------------------')

        if model == out_dir_source:
            model_labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
            model_tag = 'source'
        elif model == out_dir_target:
            model_labels_to_evaluate = [label.split('_')[1].lower() for label in target_labels_to_evaluate]
            model_tag = 'target'

        print(f'Evaluating {len(model_labels_to_evaluate)} labels: {model_labels_to_evaluate}')
        # input()

        # Load analyzer detection scores for each evaluation file example
        print(f'Loading {model} detection scores for evaluation examples...')
        score_files = []
        score_files.extend(files.find_files(model, '.csv', exclude_dirs=['threshold_perf'])) 
        predictions = pd.DataFrame()
        i = 0
        for file in score_files:
            if i % 100 == 0:
                print(f"{round(i/len(score_files) * 100, 2)}%")
            score = pd.read_csv(file)
            score = score.drop(score.columns.difference(['Common name', 'Confidence']), axis=1)
            score['file_audio'] = os.path.basename(file)
            if len(score) > 0:
                predictions = pd.concat([predictions, score], ignore_index=True)
            i += 1
        predictions['file_audio'] = predictions['file_audio'].apply(lambda x: x.removesuffix('.BirdNET.results.csv'))
        predictions.rename(columns={'file_audio': 'file'}, inplace=True)
        predictions.rename(columns={'Common name': 'label_predicted'}, inplace=True)
        predictions.rename(columns={'Confidence': 'confidence'}, inplace=True)
        predictions['label_predicted'] = predictions['label_predicted'].str.lower()
        
        if evaluation_dataset == 'validation': # Discard scores for files used in training
            predictions = predictions[predictions['file'].isin(set(evaluation_data['file']))]
        
        # Discard prediction scores for labels not under evaluation
        predictions = predictions[predictions['label_predicted'].isin(set(model_labels_to_evaluate))]
        predictions['label_truth'] = ''

        if model == out_dir_source:
            raw_predictions_source = predictions
        elif model == out_dir_target:
            raw_predictions_target = predictions
        
        print('Loading corresponding annotations...')
        if evaluation_dataset == 'validation': # TODO
            # Load annotation labels for the training files as dataframe with columns:
            # 'file' - the basename of the audio file
            # 'labels' - true labels, separated by ', ' token
            annotations = pd.read_csv(f'{training_data_path}/training_data_annotations.csv')
    
            # Discard annotations for files that were used in training
            print('Discarding annotations for files used in training...')
            annotations = annotations[annotations['file'].isin(set(evaluation_data['file']))]
        
        elif evaluation_dataset == 'test':
            annotations = evaluation_data.copy()
            annotations['file'] = annotations['file'].apply(lambda x: x.removesuffix('.wav'))
        
        # At this point, "predictions" contains the model confidence score for each evaluation example for each predicted label,
        # and "annotations" contains the all annotations (i.e. true labels) for each evaluation example.

        # Next, for each label, collate annotation data into simple presence ('label_predicted') or absence ('0') truth per label prediction per file, then add to 'predictions'
        print('Collating annotations for each label...')
        count = 0
        annotations_files_set = set(annotations['file'])
        for i, row in predictions.iterrows():
            if count % 1000 == 0:
                print(f"{round(count/len(predictions) * 100, 2)}%")
            count += 1

            conf = row['confidence']

            # DEBUG
            # TODO
            if row['file'] not in annotations_files_set: # TODO: AND / INSTEAD the original target for this prediction is not in class_labels...
                # print_warning('Setting invalid file to unknown')
                predictions.at[i, 'label_truth'] = 'unknown'
                # input()
                continue

            # Does the file truly contain the label?
            present = False
            unknown = False
            file_annotations = annotations[annotations['file'] == row['file']]
            true_labels = []
            if len(file_annotations) == 0:
                predictions.at[i, 'label_truth'] = '0' # NOTE: Unannotated files (e.g. Background files) are intepreted as having no relevant signals (i.e. labels) present
                continue
            
            true_labels = str(file_annotations['labels'].iloc[0]).split(', ')
            if len(true_labels) > 0:
                simple_labels = []
                for label in true_labels:
                    if label not in ['unknown', 'not_target']:
                        split = label.split('_')
                        if len(split) > 1:
                            label = split[1].lower()
                    simple_labels.append(label)
                true_labels = set(simple_labels)

            present = row['label_predicted'] in true_labels

            if present:
                predictions.at[i, 'label_truth'] = row['label_predicted']
            else:
                predictions.at[i, 'label_truth'] = '0'
                if 'unknown' in true_labels:
                    predictions.at[i, 'label_truth'] = 'unknown'
                elif 'not_target' in true_labels:
                    for j, a in file_annotations.iterrows():
                        target = a['target']
                        if len(target.split('_')) > 1:
                            target = a['target'].split('_')[1].lower()
                        if target != row['label_predicted']:
                            predictions.at[i, 'label_truth'] = 'unknown'
                            break
            
        # Interpret missing labels as absences
        if predictions['label_truth'].isna().sum() > 0:
            print(f"Intepreting {predictions['label_truth'].isna().sum()} predictions with missing labels as absences...")
            predictions['label_truth'] = predictions['label_truth'].fillna(0)

        # Drop unknown labels
        if len(predictions[predictions['label_truth'] == 'unknown']) > 0:
            print(f"Dropping {len(predictions[predictions['label_truth'] == 'unknown'])} predictions with unknown labels...")
            predictions = predictions[predictions['label_truth'] != 'unknown']

        # Use 'predictions' to evaluate performance metrics for each label
        model_performance_metrics = pd.DataFrame() # Container for performance metrics of all labels
        for label_under_evaluation in model_labels_to_evaluate:
            print(f"Calculating performance metrics for '{label_under_evaluation}'...")

            # Get all the predictions and their associated confidence scores for this label
            detection_labels = predictions[predictions['label_predicted'] == label_under_evaluation]

            species_performance_metrics = evaluate_species_performance(detection_labels=detection_labels, species=label_under_evaluation, plot=plot_precision_recall, title_label=model, save_to_dir=f'results/{target_model_stub}/sample_perf/threshold_perf_{model_tag}')
            model_performance_metrics = pd.concat([model_performance_metrics, species_performance_metrics], ignore_index=True)

        model_performance_metrics['model'] = model_tag
        performance_metrics = pd.concat([performance_metrics, model_performance_metrics], ignore_index=True)

        # Display results and save to file
        model_performance_metrics[model_performance_metrics.select_dtypes(include='number').columns] = model_performance_metrics.select_dtypes(include='number').round(2)
        model_performance_metrics['label'] = model_performance_metrics['label'].str.title()
        model_performance_metrics.loc[model_performance_metrics['N_pos'] == 0, ['PR_AUC', 'AP', 'ROC_AUC', 'f1_max']] = np.nan
        model_performance_metrics = model_performance_metrics.sort_values(by=['PR_AUC'], ascending=False).reset_index(drop=True)
        print(f'PERFORMANCE METRICS FOR {model}')
        print(model_performance_metrics.to_string())

        if model == out_dir_source:
            fp = f"results/{target_model_stub}/sample_perf/metrics_source.csv"
        elif model == out_dir_target:
            fp = f"results/{target_model_stub}/sample_perf/metrics_target.csv"
        model_performance_metrics.to_csv(fp, index=False)
        print_success(f'Results saved to {fp}')

        if plot_precision_recall:
            plt.show()

    print('FINAL RESULTS (vocalization level) ---------------------------------------------------------------------------')
    performance_metrics = performance_metrics.drop_duplicates()
    performance_metrics.sort_values(by=['PR_AUC', 'label', 'model'], inplace=True)
    performance_metrics.loc[performance_metrics['N_pos'] == 0, ['PR_AUC', 'AP', 'ROC_AUC', 'f1_max']] = np.nan
    # TODO
    # fp = f'data/cache/{target_model_stub}/test_evaluate_performance_performance_metrics.csv'
    # if not os.path.exists(f'data/cache/{target_model_stub}'):
    #     os.makedirs(f'data/cache/{target_model_stub}')
    performance_metrics.to_csv(fp, index=False)
    print_success(f'Saved performance metrics to {fp}')

    file_source_perf = f'results/{target_model_stub}/sample_perf/metrics_source.csv'
    file_target_perf = f'results/{target_model_stub}/sample_perf/metrics_target.csv'

    perf_source = pd.read_csv(file_source_perf)
    perf_source['label'] = perf_source['label'].str.lower()
    print(file_source_perf)

    perf_target = pd.read_csv(file_target_perf)
    perf_target['label'] = perf_target['label'].str.lower()
    print(file_target_perf)

    perf_combined = pd.merge(
        perf_source[['label', 'PR_AUC']].rename(columns={'PR_AUC': 'PR_AUC_source'}),
        perf_target[['label', 'PR_AUC']].rename(columns={'PR_AUC': 'PR_AUC_target'}),
        on='label', how='outer'
    )
    perf_combined['PR_AUC_max'] = perf_combined[['PR_AUC_source', 'PR_AUC_target']].max(axis=1)
    perf_combined['PR_AUC_max_model'] = np.where(
        perf_combined['PR_AUC_source'] == perf_combined['PR_AUC_max'], 'source',
        np.where(perf_combined['PR_AUC_target'] == perf_combined['PR_AUC_max'], 'target', 'source')
    )

    fp = f'results/{target_model_stub}/sample_perf/metrics_combined.csv'
    perf_combined.to_csv(fp, index=False)
    print_success(f'Saved combined performance results to {fp}')

    # Calculate metric deltas between custom and pre-trained
    if len(models) > 1:
        # TODO: Ensure these are comparing the same shared labels?
        print('Deltas between custom and pre-trained:')
        metrics_target = performance_metrics[performance_metrics['model'] == 'target'][[
            'label', 'AP', 'PR_AUC', 'ROC_AUC', 'f1_max'
        ]].rename(columns={
            'AP': 'AP_custom', 'PR_AUC': 'PR_AUC_custom', 'ROC_AUC': 'ROC_AUC_custom', 'f1_max': 'f1_max_custom'
        })
        metrics_source = performance_metrics[performance_metrics['model'] == 'source'][[
            'label', 'AP', 'PR_AUC', 'ROC_AUC', 'f1_max'
        ]].rename(columns={
            'AP': 'AP_pre_trained', 'PR_AUC': 'PR_AUC_pre_trained', 'ROC_AUC': 'ROC_AUC_pre_trained', 'f1_max': 'f1_max_pre_trained'
        })
        delta_metrics = pd.merge(metrics_target, metrics_source, on='label')
        delta_metrics['AP_Δ']      = delta_metrics['AP_custom'] - delta_metrics['AP_pre_trained']
        delta_metrics['PR_AUC_Δ']  = delta_metrics['PR_AUC_custom'] - delta_metrics['PR_AUC_pre_trained']
        delta_metrics['ROC_AUC_Δ'] = delta_metrics['ROC_AUC_custom'] - delta_metrics['ROC_AUC_pre_trained']
        delta_metrics['f1_max_Δ']  = delta_metrics['f1_max_custom'] - delta_metrics['f1_max_pre_trained']
        delta_metrics = delta_metrics.sort_index(axis=1)
        col_order = ['label'] + [col for col in delta_metrics.columns if col != 'label']
        delta_metrics = delta_metrics[col_order]
        delta_metrics = delta_metrics.drop_duplicates()
        delta_metrics = delta_metrics.sort_values(by=['PR_AUC_Δ'], ascending=False).reset_index(drop=True)

        # Calculate macro-averaged metrics for each model
        mean_values = delta_metrics.drop(columns='label').mean()
        mean_row = pd.Series(['MEAN'] + mean_values.tolist(), index=delta_metrics.columns)
        delta_metrics = pd.concat([delta_metrics, pd.DataFrame([mean_row])], ignore_index=True)

        # Format results
        delta_metrics[delta_metrics.select_dtypes(include='number').columns] = delta_metrics.select_dtypes(include='number').round(2)
        delta_metrics['label'] = delta_metrics['label'].str.title()

        print(delta_metrics)
        fp = f"results/{target_model_stub}/sample_perf/metrics_summary.csv"
        delta_metrics.to_csv(fp, index=False)
        print_success(f'Summary results saved to {fp}')