# Construct a (sample level) confusion matrix of all incorrectly labeled species predictions above
# a score threhold (e.g. >= 0.5) to quantify frequency of errors across all biotic and abiotic labels.
#
# Input:
# - Name stub of target model to evaluate from directory "models/target" (e.g. "OESF_1.0")
# - Path to the directory containing all prediction scores for the evaluation dataset (e.g. "data/interim/OESF_1.0/sample_perf/source")
# - Table containing evaluation dataset annotations (e.g. "data/test/test_data_annotations.csv")
#
# Output:
# - Confusion matrix for all incorrect predictions
#
# Afterwards, visualize results via hierarchical edge bundling with figs/fig_confusion.R
#
# User-defined parameters:
threshold = 0.5
target_model_stub = 'OESF_1.0'
model_tag = 'source' # or 'target
#############################################

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from misc import files
from misc.log import *

model = f'data/interim/{target_model_stub}/sample_perf/{model_tag}'
out_dir = f'results/{target_model_stub}/sample_perf/confusion_matrix'

# Load analyzer prediction scores for each evaluation file example
print(f'Loading "{model_tag}" model prediction scores for evaluation examples...')
score_files = []
score_files.extend(files.find_files(model, '.csv', exclude_dirs=['threshold_perf'])) 
predictions = pd.DataFrame()
i = 0
for file in score_files:
    if i % 100 == 0:
        print(f"{round(i/len(score_files) * 100, 2)}%")
    score = pd.read_csv(file)
    score = score.drop(score.columns.difference(['Common name', 'Confidence']), axis=1)
    score['file'] = os.path.basename(file)
    if len(score) > 0:
        predictions = pd.concat([predictions, score], ignore_index=True)
    i += 1

predictions['file'] = predictions['file'].apply(lambda x: x.removesuffix('.BirdNET.results.csv'))
predictions.rename(columns={'Common name': 'label_predicted'}, inplace=True)
predictions.rename(columns={'Confidence': 'confidence'}, inplace=True)
predictions['label_predicted'] = predictions['label_predicted'].str.lower()
predictions = predictions[predictions['confidence'] >= threshold] # threshold predictions
predictions = predictions.sort_values(by='file').reset_index(drop=True)
print(predictions.head().to_string())

# Retrieve raw annotation data
annotations = pd.read_csv('data/test/test_data_annotations.csv')
annotations['labels'] = annotations['labels'].fillna('')
annotations['file'] = annotations['file'].apply(lambda x: x.removesuffix('.wav'))
print(annotations.head().to_string())

# Determine which predictions are incorrect
incorrect_predictions = pd.DataFrame()
count = 0
for i, row in predictions.iterrows():
    if count % 1000 == 0:
        print(f"{round(count/len(predictions) * 100, 2)}%")
    count += 1

    print("Prediction ----")
    print(f"file: {row['file']}")
    print(f"label_predicted: {row['label_predicted']}")
    print(f"confidence: {row['confidence']}")

    if (row['confidence'] < threshold):
        continue # Skip predictions below threshold

    corresponding_annotations = annotations[annotations['file'] == row['file']]
    true_labels = str(corresponding_annotations['labels'].iloc[0]).split(', ')
    if len(true_labels) > 0:
        simple_labels = []
        for label in true_labels:
            if label not in ['unknown', 'not_target']:
                split = label.split('_')
                if len(split) > 1:
                    label = split[1].lower()
            simple_labels.append(label)
        true_labels = set(simple_labels)

    # Skip predictions that are correct or unknown
    present = row['label_predicted'] in true_labels
    if present:
        predictions.at[i, 'label_truth'] = row['label_predicted']
        continue # Skip correct prediction
    else:
        if 'unknown' in true_labels:
            continue # Skip unknown prediction
        elif 'not_target' in true_labels:
            for j, a in corresponding_annotations.iterrows():
                target = a['target']
                if len(target.split('_')) > 1:
                    target = a['target'].split('_')[1].lower()
                if target != row['label_predicted']:
                    continue # Skip unknown prediction

    # Any predictions at this point are incorrect

    # Re-label specific annotations
    true_labels = ["other poor snr" if s in ["not_target", "abiotic ambience", ""] else s for s in true_labels]
    true_labels = ["other truncation" if s in ["artifact truncation"] else s for s in true_labels]

    incorrect_predictions_to_add = pd.DataFrame(columns=["label_predicted", "label_truth"])
    for true_label in true_labels:
        incorrect_predictions_to_add = pd.concat([incorrect_predictions_to_add, pd.DataFrame({"label_predicted": [row['label_predicted']], "label_truth": [true_label]})], ignore_index=True)

    incorrect_predictions = pd.concat([incorrect_predictions, incorrect_predictions_to_add])

print(incorrect_predictions)

class_labels = sorted(pd.unique(incorrect_predictions[['label_predicted', 'label_truth']].values.ravel()))
print(class_labels)

confusion_mtx = confusion_matrix(incorrect_predictions['label_truth'], incorrect_predictions['label_predicted'], labels=class_labels)
df_confusion_mtx = pd.DataFrame(confusion_mtx, class_labels, class_labels)

sn.set_theme(font_scale=0.5)
cm = sn.heatmap(confusion_mtx, annot=True, vmin=0, vmax=5, cmap="Reds", xticklabels=class_labels, yticklabels=class_labels)
cm.set_xlabel("Predicted label")
cm.set_ylabel("True label")
plt.show()

# Export data for hierarchical edge bundling plot in R
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
confusion_matrix_filepath = f'{out_dir}/confusion_matrix_T{threshold}.csv'
df_confusion_mtx.to_csv(confusion_matrix_filepath, index=True)
print_success(f'Saved confusion matrix to {confusion_matrix_filepath}')