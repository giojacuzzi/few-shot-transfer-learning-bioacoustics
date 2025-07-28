##########################################################################################
# Evaluate site level performance metrics across habitat strata factors (early seral stand
# initiation, mid seral competitive exclusion, mid seral thinned, and late seral mature)
# for each model, including all species and only shared species.
#
# Input:
# - Name stub of target model to evaluate from directory "models/target" (e.g. "OESF_1.0")
# - Site level performance metrics and species richness estimates across thresholds (e.g. "results/OESF_1.0/test/site_perf/source/site_perf_source.csv")
# - Site key associating site IDs, ARU serialnos, and habitat strata ("data/test/site_metadata.csv")
# - Site presence-absence data ("data/test/site_presence_absence.csv")
#
# Output:
# - Pearson's correlation coefficients between alpha diversity (species richness), numerically encoded structural complexity level, and FPR, FNR, and total error
# - Site level performance metrics across habitat strata for each model and set of species labels (Table A.3).
#
# User-defined parameters:
target_model_stub  = 'OESF_1.0' # Name of the target model to evaluate from directory "models/target/{target_model_stub}"; e.g. 'custom_S1_N100_LR0.001_BS10_HU0_LSFalse_US0_I0' or None to only evaluate pre-trained model
##########################################################################################

import ast
import numpy as np
import os
import pandas as pd
from misc.log import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

print('=' * os.get_terminal_size().columns)
print('Begin site performance evaluation by factor\n')

site_metadata = pd.read_csv('data/test/site_metadata.csv')
print('Site metadata:')
print(site_metadata)

# Site-level stratum with the most site errors using minimum error threshold, both models

path_site_perf_source = f'results/{target_model_stub}/test/site_perf/source/site_perf_source.csv'
site_perf_source = pd.read_csv(path_site_perf_source)
site_perf_source = site_perf_source[site_perf_source['threshold'] == '0.8'] # threshold to maximize accuracy

path_site_perf_target = f'results/{target_model_stub}/test/site_perf/target/site_perf_target.csv'
site_perf_target = pd.read_csv(path_site_perf_target)
site_perf_target = site_perf_target[site_perf_target['threshold'] == '0.9'] # threshold to maximize accuracy

## Quantify relationship between alpha diversity (species richness), numerically encoded structural complexity level, and FPR, FNR, and total error

site_data = pd.DataFrame({
    'site': site_metadata['site'],
    'stratum': site_metadata['stratum']
})

# Calculate site-level species richness (alpha diversity)
site_presence_absence = pd.read_csv('data/test/site_presence_absence.csv')
species_df = site_presence_absence.iloc[2:].set_index(site_presence_absence.iloc[2:, 0]).iloc[:, 1:].T
species_df = species_df.apply(pd.to_numeric, errors='coerce')
richness = (species_df > 0).sum(axis=1)
richness_df = pd.DataFrame({
    'site': species_df.index,
    'richness': richness.values
})
print(richness_df)

site_data = site_data.merge(richness_df, on='site', how='left')

# Numerically encode stratum variable as increasing levels of structural complexity
site_data['stratum_encoded'] = site_data['stratum'].map({
    'stand init': 1,
    'comp excl': 2,
    'thinned': 3,
    'mature': 4
})

for model in ['source_all', 'target_all', 'source', 'target']:
    print('-' * os.get_terminal_size().columns)
    print(f'Evaluating model {model}...\n')

    if model == 'source_all':
        site_perf = site_perf_source.reset_index()
    elif model == 'target_all':
        site_perf = site_perf_source[~site_perf_source['label'].isin(site_perf_target['label'])]
        site_perf = pd.concat([site_perf, site_perf_target], ignore_index=True)
    elif model == 'source':
        site_perf = site_perf_source[site_perf_source['label'].isin(site_perf_target['label'])].reset_index()
    elif model == 'target':
        site_perf = site_perf_target.reset_index()

    model_site_data = site_data.copy()
    for col in ['TP', 'FP', 'TN', 'FN']: # initialize counts per site
        model_site_data[col] = 0

    def parse_sites(cell):
        if isinstance(cell, str):
            return cell.strip("[]").replace("'", "").split()
        return []

    # Iterate over species to count instances of FP, TP, FN, and TN
    for _, row in site_perf.iterrows():
        sites_detected = parse_sites(row['sites_detected'])
        sites_notdetected = parse_sites(row['sites_notdetected'])
        sites_valid = set(parse_sites(row['sites_valid']))
        sites_error = set(parse_sites(row['sites_error']))

        for idx, site in model_site_data['site'].items():
            if site not in sites_valid:
                continue # the true presence/absence of the species at this site is unknown

            if site in sites_detected:
                if site in sites_error:
                    model_site_data.at[idx, 'FP'] += 1
                else:
                    model_site_data.at[idx, 'TP'] += 1
            elif site in sites_notdetected:
                if site in sites_error:
                    model_site_data.at[idx, 'FN'] += 1
                else:
                    model_site_data.at[idx, 'TN'] += 1

    # Calculate false positive rate, false negative rate, and total error rate
    model_site_data['FPR'] = model_site_data['FP'] / (model_site_data['FP'] + model_site_data['TN'])
    model_site_data['FNR'] = model_site_data['FN'] / (model_site_data['FN'] + model_site_data['TP'])
    model_site_data['TE'] = (model_site_data['FP'] + model_site_data['FN']) / (model_site_data['TP'] + model_site_data['FP'] + model_site_data['TN'] + model_site_data['FN'])

    # Quantify relationships between predictors (richness and strucutral complexity) and responses (error metrics)
    for predictor in ['richness', 'stratum_encoded']:
        site_data_sorted = model_site_data.sort_values(predictor)

        for metric in ['FPR', 'FNR', 'TE']:
            metric_data = site_data_sorted[[metric, predictor]].dropna()
            r, p = pearsonr(metric_data[predictor], metric_data[metric])
            print(f"{predictor} x {metric}:")
            print(f" Pearson's r = {r:.3f}, p = {p:.3g}")

        plt.scatter(site_data_sorted[predictor], site_data_sorted['FPR'], color='blue', marker='o', label='FPR')
        plt.scatter(site_data_sorted[predictor], site_data_sorted['FNR'], color='green', marker='o', label='FNR')
        plt.scatter(site_data_sorted[predictor], site_data_sorted['TE'], color='red', marker='o', label='TE')
        plt.xlabel(predictor)
        plt.ylabel('error rate')
        plt.title(model)
        plt.legend()
        plt.grid(True)
        plt.show()

## Generate Table A.3, mean error rates across the gradient of habitat types, structural complexity, and alpha diversity

def get_list_from_string(s):
    return(ast.literal_eval(s.replace(" ", ",")))

for model in ['source_all', 'target_all', 'source', 'target']:

    print('-' * os.get_terminal_size().columns)
    print(f'Evaluating model {model}...\n')

    if model == 'source_all':
        site_perf = site_perf_source.reset_index()
    elif model == 'target_all':
        site_perf = site_perf_source[~site_perf_source['label'].isin(site_perf_target['label'])]
        site_perf = pd.concat([site_perf, site_perf_target], ignore_index=True)
    elif model == 'source':
        site_perf = site_perf_source[site_perf_source['label'].isin(site_perf_target['label'])].reset_index()
    elif model == 'target':
        site_perf = site_perf_target.reset_index()

    # For each species
    species_by_stratum_site_perf = pd.DataFrame()
    for index, row in site_perf.iterrows():
        label = row['label']
        sites_error = get_list_from_string(row['sites_error'])
        sites_valid = get_list_from_string(row['sites_valid'])
        sites_detected = get_list_from_string(row['sites_detected'])
        sites_notdetected = get_list_from_string(row['sites_notdetected'])

        # For each stratum
        temp_stratum_error_rates = pd.DataFrame()
        for stratum in site_metadata['stratum'].unique():

            # Calculate error rate for this species across sites of this stratum
            stratum_sites = site_metadata[site_metadata['stratum'] == stratum]['site']

            sites_error_stratum = stratum_sites[stratum_sites.isin(sites_error)]

            sites_valid_stratum = stratum_sites[stratum_sites.isin(sites_valid)]

            sites_detected_stratum = stratum_sites[stratum_sites.isin(sites_detected)]
            sites_fp_stratum = sites_detected_stratum[sites_detected_stratum.isin(sites_error)]

            sites_notdetected_stratum = stratum_sites[stratum_sites.isin(sites_notdetected)]
            sites_fn_stratum = sites_notdetected_stratum[sites_notdetected_stratum.isin(sites_error)]

            try:
                error_rate = len(sites_error_stratum) / len(sites_valid_stratum)
            except ZeroDivisionError:
                error_rate = np.nan
            
            try:
                fp_rate = len(sites_fp_stratum) / len(sites_valid_stratum)
            except ZeroDivisionError:
                fp_rate = np.nan
            
            try:
                fn_rate = len(sites_fn_stratum) / len(sites_valid_stratum)
            except ZeroDivisionError:
                fn_rate = np.nan

            temp = pd.DataFrame([{
                'class': label,
                'stratum': stratum,
                'error_rate': error_rate,
                'fp_rate': fp_rate,
                'fn_rate': fn_rate
            }])            
            temp_stratum_error_rates = pd.concat([temp_stratum_error_rates, temp], ignore_index=True)
        
        species_by_stratum_site_perf = pd.concat([species_by_stratum_site_perf,temp_stratum_error_rates], ignore_index=True)

    print('Error rate:')
    results_error_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='error_rate')
    results_error_rates.reset_index(inplace=True)
    results_error_rates['model'] = model
    print(results_error_rates.round(2))

    print('Error rate means:')
    results_error_rates_means = results_error_rates.mean(numeric_only=True)
    print(results_error_rates_means.round(2))

    print('False positive rate:')
    results_fp_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='fp_rate')
    results_fp_rates.reset_index(inplace=True)
    results_fp_rates['model'] = model
    print(results_fp_rates.round(2))

    print('False positive rate means:')
    results_fp_rates_means = results_fp_rates.mean(numeric_only=True)
    print(results_fp_rates_means.round(2))

    print('False negative rate:')
    results_fn_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='fn_rate')
    results_fn_rates.reset_index(inplace=True)
    results_fn_rates['model'] = model
    print(results_fn_rates.round(2))

    print('False negative rate means:')
    results_fn_rates_means = results_fn_rates.mean(numeric_only=True)
    print(results_fn_rates_means.round(2))

    print_success(f'Finished evaluating site perf by factor for model "{model}"')
