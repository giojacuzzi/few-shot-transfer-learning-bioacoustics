# Evaluate site level performance metrics across habitat strata factors (early seral stand
# initiation, mid seral competitive exclusion, mid seral thinned, and late seral mature)
# for each model, including all species and only shared species.
#
# Input:
# - Name stub of target model to evaluate from directory "models/target" (e.g. "OESF_1.0")
# - Site level performance metrics and species richness estimates across thresholds
# - Site key associating site IDs, ARU serialnos, and habitat strata ("data/site_key.csv")
#
# Output:
# - Site level performance metrics across habitat strata for each model and set of species labels.
#
# User-defined parameters:
target_model_stub  = 'OESF_1.0' # Name of the target model to evaluate from directory "models/target/{target_model_stub}"; e.g. 'custom_S1_N100_LR0.001_BS10_HU0_LSFalse_US0_I0' or None to only evaluate pre-trained model
#############################################

import pandas as pd
import numpy as np
import ast
from misc.log import *

site_key = pd.read_csv('data/site_key.csv')
print(site_key)

# Site-level stratum with the most site errors using minimum error threshold, both models

path_site_perf_source = f'results/{target_model_stub}/test/site_perf/source/site_perf_source.csv'
site_perf_source = pd.read_csv(path_site_perf_source)
site_perf_source = site_perf_source[site_perf_source['threshold'] == '0.8']
print(site_perf_source) 

path_site_perf_target = f'results/{target_model_stub}/test/site_perf/target/site_perf_target.csv'
site_perf_target = pd.read_csv(path_site_perf_target)
site_perf_target = site_perf_target[site_perf_target['threshold'] == '0.9']
print(site_perf_target.to_string())

def get_list_from_string(s):
    return(ast.literal_eval(s.replace(" ", ",")))

for model in ['source_all', 'target_all', 'source', 'target']:

    print(f'Evaluating model {model}...')

    if model == 'source_all':
        site_perf = site_perf_source.reset_index()
    elif model == 'target_all':
        site_perf = site_perf_source[~site_perf_source['label'].isin(site_perf_target['label'])]
        site_perf = pd.concat([site_perf, site_perf_target], ignore_index=True)
    elif model == 'source':
        site_perf = site_perf_source[site_perf_source['label'].isin(site_perf_target['label'])].reset_index()
    elif model == 'target':
        site_perf = site_perf_target.reset_index()
    
    print(site_perf['label'])

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
        for stratum in site_key['stratum'].unique():

            # Calculate error rate for this species across sites of this stratum
            stratum_sites = site_key[site_key['stratum'] == stratum]['site']

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

    print(f'species_by_stratum_error_rates')
    print(species_by_stratum_site_perf.to_string())

    print('ERROR RATE:')
    results_error_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='error_rate')
    results_error_rates.reset_index(inplace=True)
    results_error_rates['model'] = model

    results_error_rates_means = results_error_rates.mean(numeric_only=True)
    print(f'model "{model}" results_error_rates_means:')
    print(results_error_rates_means.round(2))

    print('FALSE POSITIVE RATE:')
    results_fp_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='fp_rate')
    results_fp_rates.reset_index(inplace=True)
    results_fp_rates['model'] = model

    results_fp_rates_means = results_fp_rates.mean(numeric_only=True)
    print(f'model "{model}" results_fp_rates_means:')
    print(results_fp_rates_means.round(2))

    print('FALSE NEGATIVE RATE:')
    results_fn_rates = species_by_stratum_site_perf.pivot(index='class', columns='stratum', values='fn_rate')
    results_fn_rates.reset_index(inplace=True)
    results_fn_rates['model'] = model

    results_fn_rates_means = results_fn_rates.mean(numeric_only=True)
    print(f'model "{model}" results_fn_rates_means:')
    print(results_fn_rates_means.round(2))

    print_success(f'Model "{model}" finished.')