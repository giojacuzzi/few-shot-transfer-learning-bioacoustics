# Evaluate site level (presence/absence) performance of a custom target model on the test dataset, and compare performance with a pre-trained source model.
#
# Input:
# - Name stub of target model to evaluate from directory "models/target" (e.g. "OESF_1.0")
# - Source and target labels lists (e.g. "models/source/source_species_list.txt" and "models/target/OESF_1.0/OESF_1.0_Labels.txt")
# - Complete sample level performance metrics ("results/{target_model_stub}/sample_perf/metrics_complete.csv")
# - Path to directory containing prediction scores from both the source and target models for the entire monitoring period under evaluation, saved to "data/interim/{target_model_stub}/site_perf/raw_predictions/{model_tag}". (Note these data are NOT produced by this script; you must generate them via process_audio or the GUI)
# - Table containing site true presence and absence for all species ("data/test/site_presence_absence.csv")
# - Site key associating site IDs, ARU serialnos, and habitat strata ("data/site_key.csv")
#
# Output:
# - Site level performance metrics and species richness estimates across thresholds

# CHANGE ME ###
target_model_stub  = 'OESF_1.0' # Name of the target model to evaluate from directory "models/target/{target_model_stub}"
threshold_to_evaluate = '0.9' # Threshold for complete site metrics
###############

import pandas as pd
from misc.log import *
from misc.files import *
from perf.perf_metrics import *
import sys

overwrite_prediction_cache = False
overwrite_metadata_cache = False

min_site_detections = 0

out_dir = f'results/{target_model_stub}/site_perf'
out_dir_source = out_dir + '/source'
out_dir_target     = out_dir + '/target'
models = [out_dir_source, out_dir_target]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_dir_source):
    os.makedirs(out_dir_source)
if not os.path.exists(out_dir_target):
    os.makedirs(out_dir_target)

# Load class labels
source_class_labels = pd.read_csv(os.path.abspath(f'models/source/source_species_list.txt'), header=None)[0].tolist()
target_class_labels = pd.read_csv(os.path.abspath(f'models/target/target_species_list.txt'), header=None)[0].tolist()

preexisting_labels_to_evaluate = source_class_labels #[l for l in target_class_labels if l in source_class_labels]
target_labels_to_evaluate = target_class_labels

print(f"{len(preexisting_labels_to_evaluate)} preexisting labels to evaluate:")
print(preexisting_labels_to_evaluate)
print(f"{len(target_labels_to_evaluate)} target labels to evaluate:")
print(target_labels_to_evaluate)

perf_metrics_and_thresholds = pd.read_csv(f'results/{target_model_stub}/sample_perf/metrics_complete.csv')

# Data culling – get minimum confidence score to retain a prediction for analysis (helps speed up analysis process considerably)
# labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
class_thresholds = perf_metrics_and_thresholds
class_thresholds['label'] = class_thresholds['label'].str.lower()
threshold_min_Tp  = min(class_thresholds['Tp'])
threshold_min_Tf1 = min(class_thresholds['Tf1'])
class_thresholds['min'] = class_thresholds.apply(lambda row: min(row['Tp'], row['Tf1'], 0.5), axis=1)
class_thresholds.loc[class_thresholds['min'] < 0.01, 'min'] = 0.5 # set missing classes to 0.5 minimum
print(class_thresholds[['label', 'Tp', 'Tf1', 'min']].to_string())
min_conf_dict = dict(zip(class_thresholds['label'], class_thresholds['min']))

def remove_extension(f):
    return os.path.splitext(f)[0]

# Load and cache analyzer prediction scores for ALL raw audio files
if overwrite_prediction_cache:
    for model in models:
        print(f'Loading {model} prediction scores for test examples...')

        if model == out_dir_source:
            score_dir_root = f'data/interim/{target_model_stub}/site_perf/raw_predictions/source'
        elif model == out_dir_target:
            score_dir_root = f'data/interim/{target_model_stub}/site_perf/raw_predictions/target'

        score_files = []
        score_files.extend(find_files(score_dir_root, '.csv')) 
        predictions = pd.DataFrame()
        i = 0
        for file in score_files:
            if i % 50 == 0:
                print(f"{round(i/len(score_files) * 100, 2)}% ({i} of {len(score_files)} files)")
            try:
                score = pd.read_csv(file, low_memory=False, usecols=['common_name', 'confidence', 'start_date']) # TODO
            except Exception as e:
                print_warning(f'{e}')
                print_warning(f'Incompatible columns in file {file}. Skipping...')
                continue
            score['common_name'] = score['common_name'].str.lower()

            # Cull unnecessary predictions below relevant confidence thresholds
            score = score[
                score.apply(lambda row: row['confidence'] >= min_conf_dict.get(row['common_name'], float('-inf')), axis=1)
            ]

            score['file'] = os.path.basename(file)
            predictions = pd.concat([predictions, score], ignore_index=True)
            i += 1
        predictions['file'] = predictions['file'].apply(remove_extension)
        predictions.rename(columns={'common_name': 'label_predicted'}, inplace=True)
        predictions['label_predicted'] = predictions['label_predicted'].str.lower()

        if model == out_dir_target:
            predictions.to_parquet(f'data/interim/{target_model_stub}/site_perf/cached_predictions/source/predictions_source.parquet')
        elif model == out_dir_target:
            predictions.to_parquet(f'data/interim/{target_model_stub}/site_perf/cached_predictions/target/predictions_target.parquet')

print('Loading target predictions from cache...')
predictions_target = pd.read_parquet(f'data/interim/{target_model_stub}/site_perf/cached_predictions/target/predictions_target.parquet')
print(f'Loaded {len(predictions_target)} predictions')
print('Loading source predictions from cache...')
predictions_source = pd.read_parquet(f'data/interim/{target_model_stub}/site_perf/cached_predictions/source/predictions_source.parquet')
print(f'Loaded {len(predictions_source)} predictions')

print(f'PERFORMANCE EVALUATION - site level ================================================================================================')

# Load site true presence and absence
print('Loading site true presence and absence...')
site_presence_absence = pd.read_csv('data/test/site_presence_absence.csv', header=None)

print('Site key:')
site_key = pd.read_csv('data/site_key.csv')
site_key['date_start'] = pd.to_datetime(site_key['date_start'], format='%Y%m%d').dt.date
site_key['date_end'] = pd.to_datetime(site_key['date_end'], format='%Y%m%d').dt.date
print(site_key)

site_presence_absence = site_presence_absence.iloc[3:].reset_index(drop=True)
site_presence_absence.set_index(0, inplace=True)
site_presence_absence.columns = site_key['site']
nan_rows = site_presence_absence[site_presence_absence.isna().any(axis=1)]  # Select rows with any NaN values
if not nan_rows.empty:
    print(f"WARNING: NaN values found. Dropping...")
    site_presence_absence = site_presence_absence.dropna()  # Drop rows with NaN values

# Calculate true species richness at each site
def sum_list(x):
    numeric_column = pd.to_numeric(x, errors='coerce') # coerce ? to NaN
    return int(numeric_column.sum())
true_species_richness = site_presence_absence.apply(sum_list)

def within_date_range(d, start, end):
    return start.date() <= d.date() <= end.date()

def get_matching_site(row):
    match = site_key[
        (site_key['serialno'] == row['serialno']) & 
        (site_key['date_start'] <= row['date'].date()) & 
        (site_key['date_end'] >= row['date'].date())
    ]
    if not match.empty:
        return match.iloc[0]['site']
    else:
        print_error(f'Could not find matching site for data {row}')
        return None

site_level_perf = pd.DataFrame()
site_level_perf_mean = pd.DataFrame()
for model in models:
    print(f'BEGIN MODEL EVALUATION {model} (site level) --------------------------------------------------------------------')

    if model == out_dir_source:
        # Find matching unique site ID for each prediction
        cpp = predictions_source.copy()
        model_labels_to_evaluate = [label.split('_')[1].lower() for label in preexisting_labels_to_evaluate]
        model_tag = 'source'
    elif model == out_dir_target:
        cpp = predictions_target.copy()
        intersection = [item for item in target_labels_to_evaluate if item in preexisting_labels_to_evaluate]
        model_labels_to_evaluate = [label.split('_')[1].lower() for label in intersection]
        model_tag = 'target'
    model_labels_to_evaluate = set(model_labels_to_evaluate)

    cpp['site'] = ''

    print('Calculate site-level performance per label...')
    
    # Caching
    if overwrite_metadata_cache:
        counter = 1
        for label in model_labels_to_evaluate: # model_labels_to_evaluate
            print(f'Caching metadata for class "{label}" predictions ({counter})...')
            counter += 1
            print('Copying relevant data...')
            predictions_for_label = cpp[cpp['label_predicted'] == label].copy()
            print('Parsing metadata...')
            metadata = predictions_for_label['file'].apply(parse_metadata_from_detection_audio_filename)
            serialnos = metadata.apply(lambda x: x[0]).tolist()
            dates = metadata.apply(lambda x: x[1]).tolist()
            times = metadata.apply(lambda x: x[2]).tolist()
            predictions_for_label['serialno'] = serialnos
            predictions_for_label['date']     = dates
            predictions_for_label['time']     = times
            predictions_for_label['date'] = pd.to_datetime(predictions_for_label['date'], format='%Y%m%d')

            print(f'Retrieving site IDs for {len(predictions_for_label)} predictions...')
            counter_siteid = 1
            for i, row in predictions_for_label.iterrows():
                counter_siteid += 1
                if counter_siteid % 200 == 0:
                    print(f"{round(counter_siteid/len(predictions_for_label) * 100, 2)}% ({counter_siteid} of {len(predictions_for_label)} files)")

                serialno = row['serialno']
                date = row['date']
                site = get_matching_site(row)
                predictions_for_label.at[i, 'site'] = site

            predictions_for_label.to_parquet(f'data/interim/{target_model_stub}/site_perf/cached_predictions/{model_tag}/predictions_for_label_{label}_{model_tag}.parquet')
    
    metrics = perf_metrics_and_thresholds[perf_metrics_and_thresholds['model'] == model_tag]
    metrics['label'] = metrics['label'].str.lower()

    counter = 1
    for label in model_labels_to_evaluate:
        print(f'Evaluating site-level performance for class "{label}" ({counter})...')
        counter += 1

        # Load predictions_for_label for this label from cache
        print(f'Retrieving {model_tag} predictions with metadata...')
        predictions_for_label = pd.read_parquet(f'data/interim/{target_model_stub}/site_perf/cached_predictions/{model_tag}/predictions_for_label_{label}_{model_tag}.parquet')

        label_metrics = metrics[metrics['label'] == label]
        Tp = label_metrics['Tp'].iloc[0]
        Tf1 = label_metrics['Tf1'].iloc[0]

        threshold_labels = [str(x) for x in [round(n, 2) for n in np.arange(0.5, 1.00, 0.05).tolist()]] #['Tp', 'Tf1', '0.5', '0.9', '0.95', 'max_Tp_0.5', 'max_Tp_0.9', 'max_Tp_0.95']
        thresholds       = [round(n, 2) for n in np.arange(0.5, 1.00, 0.05).tolist()] #[Tp, Tf1, 0.5, 0.9, 0.95, max(Tp, 0.5), max(Tp, 0.9), max(Tp, 0.95)]
        threshold_labels.extend(['Tp','Tf1'])
        thresholds.extend([Tp,Tf1])

        species_perf = pd.DataFrame()
        for i, threshold in enumerate(thresholds):
            threshold_label = threshold_labels[i]
            threshold_value = thresholds[i]

            species_perf_at_threshold = get_site_level_confusion_matrix(label, predictions_for_label, threshold, site_presence_absence, min_detections=min_site_detections)
            species_perf_at_threshold['precision'] = species_perf_at_threshold['precision'].fillna(0.0) # if precision is NaN (i.e. no TP or FP), then no positive predictions were made despite at least one presence, so precision = 0.0
            species_perf_at_threshold['model'] = model
            species_perf_at_threshold['threshold'] = threshold_label
            species_perf_at_threshold['threshold_value'] = threshold_value

            species_perf = pd.concat([species_perf, species_perf_at_threshold], ignore_index=True)

        site_level_perf = pd.concat([site_level_perf, species_perf], ignore_index=True)

    print(f'FINAL RESULTS {model_tag} (site level) ------------------------------------------------------------------------------------------------------')
    site_level_perf = site_level_perf.reindex(sorted(site_level_perf.columns), axis=1)
    if model == out_dir_source:
        fp = f'{out_dir_source}/site_perf_source.csv'
        site_level_perf[site_level_perf["model"] == out_dir_source].to_csv(fp, index=False)
    elif model == out_dir_target:
        fp = f'{out_dir_target}/site_perf_target.csv'
        site_level_perf[site_level_perf["model"] == out_dir_target].to_csv(fp, index=False)
    print_success(f'Saved site level perf for model {model_tag} to {fp}')

file_source_perf = f'{out_dir_source}/site_perf_source.csv'
file_target_perf = f'{out_dir_target}/site_perf_target.csv'

perf_source = pd.read_csv(file_source_perf)
perf_source['label'] = perf_source['label'].str.lower()
perf_source = perf_source[perf_source['threshold'] == threshold_to_evaluate]

perf_target = pd.read_csv(file_target_perf)
perf_target['label'] = perf_target['label'].str.lower()
perf_target = perf_target[perf_target['threshold'] == threshold_to_evaluate]
perf_target = perf_target[perf_target['present'] > 0]

perf_combined = pd.merge(
    perf_source[['label', 'correct_pcnt']].rename(columns={'correct_pcnt': f'accuracy_source_{threshold_to_evaluate}'}),
    perf_target[['label', 'correct_pcnt']].rename(columns={'correct_pcnt': f'accuracy_target_{threshold_to_evaluate}'}),
    on='label', how='outer'
)
perf_combined[f'accuracy_max_{threshold_to_evaluate}'] = perf_combined[[f'accuracy_source_{threshold_to_evaluate}', f'accuracy_target_{threshold_to_evaluate}']].max(axis=1)
perf_combined[f'accuracy_max_{threshold_to_evaluate}_model'] = np.where(
    perf_combined[f'accuracy_source_{threshold_to_evaluate}'] == perf_combined[f'accuracy_max_{threshold_to_evaluate}'], 'source',
    np.where(perf_combined[f'accuracy_target_{threshold_to_evaluate}'] == perf_combined[f'accuracy_max_{threshold_to_evaluate}'], 'target', 'source')
)
perf_combined.to_csv(f'{out_dir}/site_metrics_combined.csv', index=False)

print('SITE LEVEL PERF COMPARISON ==================================================================================================')

labels_to_compare = [l for l in target_labels_to_evaluate if l in preexisting_labels_to_evaluate]
labels_to_compare = [l.split('_')[1].lower() for l in labels_to_compare]
print('Labels to compare:')
print(labels_to_compare)

for threshold_label in threshold_labels:
    print_warning(f'Evaluating site-level performance for {threshold_label}...')

    threshold_results = pd.DataFrame()
    for model in models:
        print(f'Evaluating model {model}...')

        # Get the results matching the model and the threshold, model_results
        model_results = site_level_perf[(site_level_perf['threshold'] == threshold_label) & (site_level_perf['model'] == model)].copy()
        model_results = model_results[model_results['label'].isin(labels_to_compare)]
        print_exclude_cols = ['sites_detected', 'sites_notdetected', 'sites_error']
        threshold_results = pd.concat([threshold_results,model_results[['label', 'error_pcnt', 'precision', 'recall', 'fpr', 'model']]], ignore_index=True)

    threshold_results.to_csv(f'{out_dir}/threshold_results_{threshold_label}.csv', index=False)

    merged = pd.merge(threshold_results[threshold_results['model'] == out_dir_source], threshold_results[threshold_results['model'] == out_dir_target], on='label', suffixes=('_source', '_target'))
    merged['error_pcnt_Δ'] = merged['error_pcnt_target'] - merged['error_pcnt_source']
    merged['precision_Δ']  = merged['precision_target'] - merged['precision_source']
    merged['recall_Δ']     = merged['recall_target'] - merged['recall_source']
    merged['fpr_Δ']     = merged['fpr_target'] - merged['fpr_source']
    merged = merged.reindex(sorted(merged.columns), axis=1)

    mean_values = merged.select_dtypes(include='number').mean()
    # Convert the mean values to a DataFrame with the same column names
    mean_row = pd.DataFrame(mean_values).T
    mean_row['label'] = 'Mean'
    # Append the mean row to the original DataFrame
    merged = pd.concat([merged, mean_row], ignore_index=True)

    result = merged
    result[result.select_dtypes(include='number').columns] = result.select_dtypes(include='number').round(2)
    result['label'] = result['label'].str.title()
    result.insert(0, 'label', result.pop('label'))
    result = result.loc[:, ~result.columns.str.contains('model')]
    
    fp = f'{out_dir}/results_{threshold_label}.csv'
    result.to_csv(fp)
    print_success(f'Saved results to {fp}')

# SPECIES RICHNESS COMPARISON
print('SPECIES RICHNESS COMPARISON ==================================================================================================')

# For each threshold
for threshold_label in threshold_labels:
    print_warning(f'Evaluating species richness performance for {threshold_label}...')

    # For each model
    for model in models:
        print(f'Evaluating model {model}...')

        if model == out_dir_source:
            model_tag = 'source'
        elif model == out_dir_target:
            model_tag = 'target'

        # Get the results matching the model and the threshold, model_results
        model_results = site_level_perf[(site_level_perf['threshold'] == threshold_label) & (site_level_perf['model'] == model)].copy()
        print_exclude_cols = ['sites_detected', 'sites_notdetected', 'sites_error']

        # If the model under evaluation is custom target...
        if model == out_dir_target:
            
            # Get the results source_results matching the pre-trained model and threshold
            source_results = site_level_perf[(site_level_perf['threshold'] == threshold_label) & (site_level_perf['model'] == out_dir_source)].copy()

            # Replace all rows in model_results with label values NOT in the trained list with the rows for those labels in source_results
            preexisting_untrained_labels = [l for l in preexisting_labels_to_evaluate if l not in target_labels_to_evaluate]
            preexisting_untrained_labels = [l.split('_')[1].lower() for l in preexisting_untrained_labels]
            source_results = source_results[source_results['label'].isin(preexisting_untrained_labels)]
            model_results = model_results[~model_results['label'].isin(preexisting_untrained_labels)]
            model_results = pd.concat([model_results, source_results], ignore_index=True)
        
        # Calculate stats and store for later, compute stat deltas between models for each threshold
        print('Site species counts:') # Species richness comparison
        df_exploded = model_results.explode('sites_detected') # make one row per site-species detection
        site_species_counts = df_exploded.groupby('sites_detected')['label'].count() # get count of species (i.e. labels) for each site
        site_species_counts = site_species_counts.reset_index(name='species_count')

        site_species_counts['true_species_richness'] = site_species_counts['sites_detected'].map(true_species_richness)
        site_species_counts['sr_delta'] = site_species_counts['species_count'] - site_species_counts['true_species_richness']
        site_species_counts['sr_delta_pcnt'] = (site_species_counts['species_count'] / site_species_counts['true_species_richness']) * 100.0

        # Display the updated DataFrame
        fp = f'{out_dir}/{model_tag}/speciesrichness_{model_tag}_{threshold_label}.csv'
        site_species_counts.to_csv(fp)
        print_success(f'Saved site species richness counts to {fp}')

        print(f"Total average species richness percentage: {site_species_counts['sr_delta_pcnt'].mean()}")
        mean_site_perf_at_threshold = pd.DataFrame({
            "sr_delta": [site_species_counts['sr_delta'].mean()],
            "sr_delta_pcnt": [site_species_counts['sr_delta_pcnt'].mean()], # total average species richness % of truth
            "threshold":  [threshold_label],
            "model":      [model]
        })
        print(mean_site_perf_at_threshold)
        site_level_perf_mean = pd.concat([site_level_perf_mean, mean_site_perf_at_threshold], ignore_index=True)

        # Determine effect of habitat type on performance
        print('Average species richness percentage difference by strata:')
        merged_df = pd.merge(site_key, site_species_counts, left_on='site', right_on='sites_detected', how='inner')
        average_percentage_Δ_by_stratum = merged_df.groupby('stratum')['sr_delta_pcnt'].mean()
        # print('Species richness percentage difference:')
        print('average SR percent difference by stratum:')
        print(average_percentage_Δ_by_stratum)
  
print('FINAL MEAN SITE LEVEL SPECIES RICHNESS ESTIMATE:')
print(site_level_perf_mean.to_string())
fp = f'{out_dir}/speciesrichness_summary.csv'
site_level_perf_mean.to_csv(fp)
print_success(f'Saved summary species richness metrics to {fp}')
