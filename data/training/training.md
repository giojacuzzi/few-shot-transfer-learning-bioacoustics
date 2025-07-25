This directory contains data for target model training, including a subdirectory `audio` containing all audio files for training and a table `training_data_annotations.csv` containing all associated annotations for those files. Find example training data included with [the latest GitHub release](https://github.com/giojacuzzi/few-shot-transfer-learning-bioacoustics/releases). This directory is accessed by scripts `src/submodules/BirdNET-Analyzer/pretrain_stratified_kfold.py` and `src/submodules/BirdNET-Analyzer/train_fewshot.py`.

The `audio` directory may be organized with subdirectories, if desired. Regardless of organization, note that training data examples may contain multiple class signals of interest –– the actual true contents (i.e. labels) of the examples are specified in `training_data_annotations.csv`. The only required subdirectory in `audio` is `audio/Background`, which contains all examples of the ambient background noise floor (i.e. training examples with no discrete vocalizations or audio signals of interest).

The `training_data_annotations.csv` table associates annotation labels with each individual training example, and uses the example format shown below, where "audio_subdir" indicates the name of the subdirectory in which the audio file is located, "file" the name of the audio file (excluding file extension), and "labels" a comma-separated list of class labels associated with the file:

| audio_subdir | file    | labels  |
| --------- | ------- | ------- |
| Abiotic_Abiotic Aircraft | SMA00309_20200501_090127_907.6229 | "Abiotic_Abiotic Aircraft, Regulus satrapa_Golden-crowned Kinglet" |
| Abiotic_Abiotic Wind | SMA00351_20200428_080038_142.4009 | "Abiotic_Abiotic Wind, Regulus satrapa_Golden-crowned Kinglet" |
| Troglodytes pacificus_Pacific Wren | SMA00349_20200619_060002_195.34805 | "Troglodytes pacificus_Pacific Wren, Ixoreus naevius_Varied Thrush, Piranga ludoviciana_Western Tanager" |
| Strix varia_Barred Owl | SMA00424_20200521_020000_678.6368500000001 | "Strix varia_Barred Owl, Abiotic_Abiotic Rain" |
| ... | ... | ... |

