# Evaluate performance for a single class (species)

import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Plotting
import sklearn.metrics          # Classifier evaluation
import numpy as np              # Mathematics
from misc.log import *
import os

# detection_labels - a dataframe of detections with columns 'label_truth' (where a positive presence is represented by the species class label) and 'confidence'
# species - the species class label, e.g. "american crow"
def evaluate_species_performance(detection_labels, species, plot, digits=3, title_label='', save_to_dir=''):
    
    plots = []

    # Exclude files with an "unknown" label_truth from consideration
    n_unknown = (detection_labels['label_truth'] == 'unknown').sum()
    if n_unknown > 0:
        print_warning(f"{n_unknown} detections with unknown species. Excluding these from consideration...")
        detection_labels = detection_labels[detection_labels['label_truth'] != 'unknown']

    n_examples = len(detection_labels)
    n_P = sum(detection_labels['label_truth'] == species) # Total number of positive examples
    n_N = sum(detection_labels['label_truth'] != species) # Total number of negative examples
    
    if len(detection_labels) == 0:
        print_error(f"No detections to evaluate for '{species}'.")
        return

    # Precision is the proportion of true positives among positive predictions, TP/(TP + FP). Intuitively,
    # when the model says "Barred Owl", how often is it correct? Precision matters when the cost of false
    # positives is high; when misclassifying as present has serious consequences.
    #
    # Recall is the proportion of true positives identified out of all actual positives, TP/(TP + FN), also
    # called the "true positive rate" or "sensitivity". Intuitively, out of all actual Barred Owl calls, how
    # many did the model correctly identify? Recall matters when the cost of false negatives is high; when
    # misclassifying as absent has serious consequences. High recall relates to a low false negative rate.
    #
    # The Precision-Recall curve summarizes the tradeoff between precision and recall as we vary the
    # confidence threshold. This relationship is only concerned with the correct prediction of the species
    # label, and does not require true negatives to be calculated. As such, Precision-Recall curves are
    # appropriate when the evaluation data are imbalanced between each class (i.e. present vs non-present).
    # This means is that a large number of negative instances won’t skew our understanding of how well our model
    # performs on the positive species class.
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species)

    f1_scores = 2*recall*precision/(recall+precision)
    if np.any(np.isnan(f1_scores)):
        f1_scores = f1_scores[~np.isnan(f1_scores)]
    f1_max = np.max(f1_scores)

    if save_to_dir != '':
        os.makedirs(save_to_dir, exist_ok=True)
        stats = pd.DataFrame({
            'threshold': thresholds,
            'precision': precision[:-1],
            'recall': recall[:-1]
        })
        stats = stats.sort_values(by=['threshold', 'precision', 'recall'])
        stats.to_csv(f'{save_to_dir}/{species}.csv', index=False)

    padding = 0.01
    font_size = 9

    if plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # Plot precision and recall as a function of threshold
        ax1.plot(thresholds, precision[:-1], label='Precision', marker='.') #, marker='.' 
        ax1.plot(thresholds, recall[:-1], label='Recall', marker='.') # , marker='.'
        # ax1.plot(thresholds, f1_scores[:-1], label='F1 Score') # , marker='.' # TODO: plot this
        ax1.set_xlabel('Threshold') 
        ax1.set_ylabel('Performance')
        ax1.set_title(f'{title_label}\n{species}\nThreshold performance', fontsize=font_size)
        ax1.set_xlim(0.0-padding, 1.0+padding)
        ax1.set_ylim(0.0-padding, 1.0+padding)
        ax1.set_box_aspect(1)
        ax1.legend(loc='lower left') 

    # The area under the precision-recall curve, AUC-PR, is a useful summary statistic of the the relationship.
    # The AUC-PR ranges from 0.0 to 1.0, with 1.0 indicating a perfect classifier. However, unlike the AUC
    # of the ROC curve, the AUC-PR of a random (baseline) classifier is equal to the proportion of positive
    # instances in the dataset. This makes it a more conservative (and in many cases, more realistic) metric
    # when dealing with imbalanced data. Note that because this is calculated using the interpolated trapezoidal
    # rule, it can lead to an overestimated measurement of performance
    pr_auc = sklearn.metrics.auc(recall, precision)

    # The average precision from prediction scores (AP) summarizes a precision-recall curve as the weighted mean of
    # precisions achieved at each threshold, with the increase in recall from the previous threshold used as the
    # weight. This can provide a more realistic measure of performance than AUC as it is not interpolated among scores.
    pr_ap = sklearn.metrics.average_precision_score(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species)

    # A baseline or "unskilled" classifier is one that cannot discriminate between the classes and would
    # predict a random class or a constant class in all cases. This is represented by a horizontal line
    # at the ratio of positive to negative examples in the dataset.
    no_skill = len(detection_labels[detection_labels['label_truth']==species]) / len(detection_labels)

    if plot:
        # Plot precision-recall curve
        ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Baseline', color='gray')
        ax2.plot(recall, precision, label='Classifier') #  marker='.',
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{title_label}\n{species}\nPrecision-Recall (AUC {pr_auc:.2f}, AP {pr_ap:.2f})', fontsize=font_size)
        ax2.set_xlim([0.0-padding, 1.0+padding])
        ax2.set_ylim([0.0-padding, 1.0+padding])
        ax2.legend(loc='lower left')
        ax2.set_box_aspect(1)

    # Plot ROC
    if True:
        # NOTE: ROC EVALUATIONS SHOWN FOR DEMONSTRATION. IMBALANCED EVALUATION DATA PRECLUDES INFERENCE.
        # The Receiver Operating Characteristic curve summarizes the tradeoff between the true positive rate (i.e. recall)
        # and the false positive rate (FPR) as we vary the confidence threshold. ROC curves are appropriate when the
        # evaluation data are balanced between each class (i.e. present vs non-present).
        fpr, tpr, roc_th = sklearn.metrics.roc_curve(detection_labels['label_truth'], detection_labels['confidence'], pos_label=species, drop_intermediate=False)

        # The ROC Area Under the Curve (ROC-AUC) is a useful summary statistic of the ROC curve that tells us the
        # probability that a randomly-selected positive example ranks higher than a randomly-selected negative one.
        # ROC-AUC == 1 if our model is perfect, while ROC-AUC == 0.5 if our model is no better than a coin flip.
        # Note that a low ROC-AUC can also reflect a class imbalance in the evaluation data.
        if n_P > 0 and n_N > 0:
            roc_auc = sklearn.metrics.roc_auc_score(detection_labels['label_truth'], detection_labels['confidence'])
        else:
            print_warning(f"Could not compute ROC AUC ({n_P} positive, {n_N} negative examples).")
            roc_auc = np.NaN

        # if plot:
        #     # ns_probs = [species for _ in range(len(detection_labels))] # no skill classifier that only predicts 1 for all examples
        #     # ns_roc_auc_score = sklearn.metrics.roc_auc_score(detection_labels['label_truth'], ns_probs, pos_label=species)
        #     # ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(detection_labels['label_truth'], ns_probs, pos_label=species, drop_intermediate=False)
        #     # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline', color='gray')
        #     ax3.plot(fpr, tpr, marker='.', label='Classifier')
        #     ax3.set_xlabel('False Positive Rate')
        #     ax3.set_ylabel('True Positive Rate (Recall)')
        #     ax3.set_title(f'ROC (AUC {roc_auc:.2f})', fontsize=font_size)
        #     ax3.set_xlim([0.0-padding, 1.0+padding])
        #     ax3.set_ylim([0.0-padding, 1.0+padding])
        #     ax3.legend(loc='lower right')

    if plot:
        # fig.suptitle(species, x=0.0, y=1.0, horizontalalignment='left', verticalalignment='top', fontsize=12)
        plt.tight_layout()
        plots.append((fig, (ax1, ax2)))
    
    if plot:
        for fig, ax in plots:
            fig.show()
            plt.show(block=False)

    if 'site' in detection_labels.columns:
        N_sites = detection_labels[detection_labels['label_truth'] == species]['site'].nunique(dropna=True)
        sites = sorted(set(detection_labels[detection_labels['label_truth'] == species]['site'].dropna().tolist()))
    else:
        N_sites = np.nan
        sites = []

    # print('fpr')
    # print(fpr)
    # print('roc_th')
    # print(roc_th)
    # input()
    
    # Get index of value in array, or nearest greater value if it doesn't exit 
    def argnearest(arr, v):
        idx_exact_match = np.where(arr == v)[0]
        if idx_exact_match.size > 0:
            return idx_exact_match[0]
        idx_gt = np.where(arr > v)[0]
        if idx_gt.size > 0:
            return idx_gt[np.argmin(arr[idx_gt])]
        else:
            return None

    idx_Tp = np.argmax(precision[:-1])
    Tp     = thresholds[idx_Tp]          # Score threshold to maximize precision, Tp
    p_Tp   = precision[idx_Tp]           # Precision at Tp
    r_Tp   = recall[idx_Tp]              # Recall at Tp
    fpr_Tp = fpr[argnearest(roc_th, Tp)] # False positive rate at Tp

    idx_Tf1 = np.argmax(f1_scores)
    Tf1 = thresholds[idx_Tf1]               # Score threshold to maximize F1 score, Tf1
    p_Tf1 = precision[idx_Tf1]              # Precision at Tf1
    r_Tf1 = recall[idx_Tf1]                 # Recall at Tf1
    fpr_Tf1 = fpr[argnearest(roc_th, Tf1)]  # False positive rate at Tf1

    idx_T095 = argnearest(thresholds, 0.95)
    if idx_T095 != None:
        p_T095 = precision[idx_T095] # Precision at naive aribtary threshold 0.9
        r_T095 = recall[idx_T095]    # Recall at naive aribtary threshold 0.9
        fpr_T095 = fpr[argnearest(roc_th, 0.95)]
    else:
        p_T095 = 0.0
        r_T095 = 0.0

    idx_T09 = argnearest(thresholds, 0.9)
    if idx_T09 != None:
        p_T09 = precision[idx_T09] # Precision at naive aribtary threshold 0.9
        r_T09 = recall[idx_T09]    # Recall at naive aribtary threshold 0.9
        fpr_T09 = fpr[argnearest(roc_th, 0.9)]
    else:
        p_T09 = 0.0
        r_T09 = 0.0
        fpr_T09 = 0.0

    idx_T05 = argnearest(thresholds, 0.5)
    if idx_T05 != None:
        p_T05 = precision[idx_T05] # Precision at naive aribtary threshold 0.5
        r_T05 = recall[idx_T05]    # Recall at naive aribtary threshold 0.5
        fpr_T05 = fpr[argnearest(roc_th, 0.5)]
    else:
        p_T05 = 0.0
        r_T05 = 0.0
        fpr_T05 = 0.0
    
    # Return the performance metrics
    return pd.DataFrame({
        'label':   [species],
        # Summary metrics
        'PR_AUC':    [pr_auc],                                 # Precision-Recall AUC
        'AP':        [pr_ap],                                  # Average precision
        'ROC_AUC':   [roc_auc],                                # Receiver Operating Characteristic AUC
        'f1_max':    [f1_max],                                                # Maximum F1 score across all thresholds
        'conf_max': [detection_labels['confidence'].max()],    # Maximum confidence score
        # 'p_mean':    [round(precision.mean(), digits)],                       # Average precision across all thresholds
        # 'p_max':     [round(precision[np.argmax(precision[:-1])],  digits)],  # Maximum precision across all thresholds
        # Optimize for precision
        'Tp':        [Tp],     # Score threshold to maximize precision, Tp
        'p_Tp':      [p_Tp],   # Precision at Tp
        'r_Tp':      [r_Tp],   # Recall at Tp
        # 'fpr_Tp':    [fpr_Tp], # False positive rate at Tp
        # Optimize for F1 score
        'Tf1':       [Tf1],     # Score threshold to maximize F1 score, Tf1
        'p_Tf1':     [p_Tf1],   # Precision at Tf1
        'r_Tf1':     [r_Tf1],   # Recall at Tf1
        # 'fpr_Tf1':   [fpr_Tf1], # False positive rate at Tf1
        # Naive arbitrary thresholds
        'p_T095': [p_T095], # 0.95
        'r_T095': [r_T095],
        'p_T09': [p_T09], # 0.9
        'r_T09': [r_T09],
        # 'fpr_T09': [fpr_T09],
        'p_T05': [p_T05], # 0.5
        'r_T05': [r_T05],
        # 'fpr_T05': [fpr_T05],
        # Sample sizes
        'N':           [n_examples],                                            # Total number of examples (not including "unknown" examples)
        'N_pos':       [n_P],                                                   # Total number of positive examples
        'N_neg':       [n_N],                                                   # Total number of negative examples
        'N_unk':       [n_unknown],                                             # Total number of unknown examples excluded from evaluation
        'class_ratio': [round(n_P / n_examples, 2)]                           # Class balance ratio (0.5 is perfectly balanced, 0.0 only negative, 1.0 only positive)
    })

# Returns a dataframe containing a confusion matrix (TP, FP, FN, TN) and number of truly present/absent sites for a given species from a detection history
def get_site_level_confusion_matrix(species, detections, threshold, site_presence_absence, min_detections=0, precision=3):

    debug = False
    # print(detections)
    # print(f"threshold {threshold}")
    # input()

    # print(site_presence_absence)
    all_sites = site_presence_absence.columns
    # print(f"all_sites ({len(all_sites)}) {all_sites}")

    species_row = site_presence_absence.loc[species]
    # print(species_row)
    # Get site names for each condition (1, ?, 0)
    sites_present = species_row[species_row == '1'].index.tolist()
    sites_unknown = species_row[species_row == '?'].index.tolist()
    sites_absent  = species_row[species_row == '0'].index.tolist()
    # Display results
    # print(f"Sites with 'present' for {species}:", sites_present)
    # print(f"Sites with 'unknown' for {species}:", sites_unknown)
    # print(f"Sites with 'absent'  for {species}:", sites_absent)

    known_sites = np.setdiff1d(all_sites, sites_unknown)
    # print(f"Valid sites: {valid_sites}")
    # input()
    
    # Filter for species detections, excluding unknown examples
    # TODO: replace
    # detections_species = detections[(detections['label_predicted'] == species)]
    # print(detections_species)

    if len(sites_present) + len(sites_unknown) + len(sites_absent) != len(all_sites):
        print_error(f'Number of sites present ({len(sites_present)}), unknown ({len(sites_unknown)}), and absent ({len(sites_absent)}) does not equal total number of sites ({len(all_sites)})')

    def truncate_float(n, places):
        return int(n * (10 ** places)) / 10 ** places

    # Sites detected using the threshold
    detections_thresholded = detections[(detections['confidence'] >= truncate_float(threshold, precision))]
    # # print('detections_thresholded')
    # # print(detections_thresholded)

    # # Post-process: only retain sites for which a minimum number of detections were made
    # if not detections_thresholded.empty:

    #     site_counts = detections_thresholded['site'].value_counts()
    #     # print(site_counts)
    #     detections_above_min_count = site_counts[site_counts > min_detections].index
    #     detections_thresholded = detections_thresholded[detections_thresholded['site'].isin(detections_above_min_count)]
    # print('after min_detections post process:')
    # print(detections_thresholded)
    # input()

    sites_detected = detections_thresholded['site'].unique()

    sites_detected = np.intersect1d(known_sites, sites_detected)

    if debug:
        print('valid_sites')
        print(known_sites)
        print('sites_detected')
        print(sites_detected)    

    # print(sites_detected)
    # print(f'Sites detected with threshold {threshold}  ({len(sites_detected)}): {sites_detected}')

    # Sites not detected using the threshold
    sites_notdetected = np.setdiff1d(known_sites, sites_detected)
    if debug:
        print(f'Sites not detected with threshold {threshold}  ({len(sites_notdetected)}): {sites_notdetected}')

    # if len(sites_detected) + len(sites_notdetected) != len(all_sites):
    #     print_error(f'Number of sites detected ({len(sites_detected)}) and not detected ({len(sites_notdetected)}) does not equal total number of sites ({len(all_sites)})')

    # TP - Number of sites correctly detected (at least once)
    tp_sites = np.intersect1d(sites_present, sites_detected)
    nsites_tp = len(tp_sites)

    # FP - Number of sites incorrectly detected (i.e. not correctly detected at least once)
    fp_sites = np.setdiff1d(np.intersect1d(sites_absent, sites_detected), tp_sites) # remove sites where the species was otherwise correctly detected at least once
    nsites_fp = len(fp_sites)

    # TN - Number of sites correctly not detected
    tn_sites = np.intersect1d(sites_notdetected, sites_absent)
    nsites_tn = len(tn_sites)

    # FN - Number of sites incorrectly not detected
    fn_sites = np.intersect1d(sites_notdetected, sites_present)
    nsites_fn = len(fn_sites)
    if debug:
        print(f'nsites_tp {nsites_tp}: {tp_sites}')
        print(f'nsites_fp {nsites_fp}: {fp_sites}')
        print(f'nsites_tn {nsites_tn}: {tn_sites}')
        print(f'nsites_fn {nsites_fn}: {fn_sites}')

    # nsites_accounted_for = nsites_tp + nsites_fp + nsites_tn + nsites_fn
    # if nsites_accounted_for != len(all_sites):
    #     print_error(f'Only {nsites_accounted_for} sites accounted for of {len(all_sites)} total')
    
    if nsites_tp + nsites_fn != len(sites_present):
        print_error(f'Incorrect true presences TP {nsites_tp} + FN {nsites_fn} != {len(sites_present)}')
    if nsites_fp + nsites_tn != len(sites_absent):
        print_error(f'Incorrect true absences FP {nsites_fp} + TN {nsites_tn} != {len(sites_absent)}')
    
    try:
        precision = nsites_tp / (nsites_tp + nsites_fp)
    except ZeroDivisionError:
        precision = 1.0
    try:
        recall = nsites_tp / (nsites_tp + nsites_fn)
    except ZeroDivisionError:
        recall = np.nan

    # False positive rate
    try:
        fpr = nsites_fp / (nsites_fp + nsites_tn)
    except ZeroDivisionError:
        fpr = 0.0 # or np.nan

    if debug:
        print(f'precision {precision}')
        print(f'recall {recall}')
        # input()
    
    result = {
        'label':          [species],
        'present':        [len(sites_present)],
        'absent':         [len(sites_absent)],
        'unknown':        [len(sites_unknown)],
        'threshold':      [threshold],
        'detected':       [len(sites_detected)],
        'notdetected':    [len(sites_notdetected)],
        'correct':        [nsites_tp + nsites_tn],
        'correct_pcnt':   [round((nsites_tp + nsites_tn) / (len(sites_present) + len(sites_absent)), 2)],
        'error':          [nsites_fp + nsites_fn],
        'error_pcnt':     [round((nsites_fp + nsites_fn) / (len(sites_present) + len(sites_absent)), 2)],
        'FP': [nsites_fp],
        'FN': [nsites_fn],
        'TP': [nsites_tp],
        'TN': [nsites_tn],
        # 'TP_pcnt': [tp_pcnt],
        # 'TN_pcnt': [tn_pcnt],
        'fpr': [fpr],
        'precision': [round(precision,3)],
        'recall':    [round(recall,3)],
        'sites_valid': [known_sites],
        'sites_detected': [sites_detected],
        'sites_notdetected': [sites_notdetected],
        'sites_error': [np.concatenate((fp_sites, fn_sites))]
    }
    return(pd.DataFrame(result, index=None))
