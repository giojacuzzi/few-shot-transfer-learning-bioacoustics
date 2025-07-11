# few-shot-transfer-learning-bioacoustics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15694435.svg)](https://doi.org/10.5281/zenodo.15694435)

This repository contains the open source software, workflow routines, and reproducible methods used in the research article:

> Jacuzzi, G., Olden, J.D., 2025. Few-shot transfer learning enables robust acoustic monitoring of wildlife communities at the landscape scale. Ecological Informatics 90, 103294. [doi.org/10.1016/j.ecoinf.2025.103294](https://doi.org/10.1016/j.ecoinf.2025.103294)

Although this software was developed for a particular study region and set of monitoring objectives (avian biodiversity surveys in the Olympic Experimental State Forest of Washington, USA), it is designed to be freely repurposed and we encourage its use in other applications. Please cite the original publication in your references. Direct any correspondance to gioj@uw.edu, and request features or bug fixes via GitHub issues.

## Quickstart: GUI application for audio analysis

Download the latest release assets via [the GitHub repository](https://github.com/giojacuzzi/few-shot-transfer-learning-bioacoustics/releases), unzip, and then launch the `Model Ensemble Interface` executable.

With the application window open, configure an analysis and/or segment process by drag-and-dropping files or directories and adjusting the interface parameters. A configuration can be saved as a `.json` *session* file (`File > Save session`), and opened for repeated use (`File > Open session`). Click the `Launch process` button to begin your analysis, and monitor progress via the console log.

# Model development and evaluation

### Recommendations
- Visual Studio Code with Microsoft extensions [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy), and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- Install dependencies to a [virtual python environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (.venv) that is used exclusively for this project. Create the environment manually, or via VS Code.

### Prerequisites
- Windows 7+ or MacOS 10.12+ operating system (64-bit). Note that the version of TensorFlow currently used alongside BirdNET is not natively compatible with Apple Silicon.
- [Python](https://www.python.org/downloads/) 3.9+ 64-bit (3.10 recommended, ensure "Add path to environment variables" is checked during install)

### Package dependencies
From a terminal shell within the virtual environment, navigate to the root directory of this repository (`few-shot-transfer-learning-bioacoustics`), and run:

```
git submodule update --init --recursive
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required dependencies, including a custom fork of BirdNET (version 2.4) with additional tools for model training and development. For troubleshooting reference, see setup instructions for [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer).

## Contents
- `data` – data associated with the study area and monitoring project
- `models` - files for source, target, and ensemble models, including associated class labels lists 
- `src` – all source code for data annotation, processing, and analysis
    - `figures` – R scripts to generate figure plots and data visualizations
    - `gui` – supporting files for the GUI
    - `misc` – helper modules and logging
    - `perf` – functions for calculating raw sample and site level performance metrics
    - `submodules` – repository dependencies (a customized fork of BirdNET-Analyzer)

## Audio classification with BirdNET and/or custom model
Run `src/process_audio.py` to process predictions for a given directory or file with a particular model. Show arguments with `python src/process_audio.py -h`, or see documentation within the script itself.

Alternatively, run the graphical user interface application with `src/gui.py`.

### GUI application packaging
To build and package the GUI as an executable from source, first activate the virtual environment from the console, then run `pyinstaller gui.spec --clean`.

## Few-shot transfer learning and performance evaluation pipeline

### Model training
*To train a custom model*, a directory containing all training audio files and a table containing all training data annotations are required. See `training/training.md` for required formatting. Adjust the user-defined parameters in `src/submodules/BirdNET-Analyzer/pretrain_stratified_kfold.py`, then run the script to generate a table of development file references for training and validation datasets. This script will output command(s) to subsequently execute via `src/submodules/BirdNET-Analyzer/train_fewshot.py` to conduct training for your model(s).

### Model performance evaluation
*To evaluate model performance*, a directory containing all evaluation (i.e. validation or test dataset) audio files and a table containing all evaluation data annotations are required. See `test/test.md` for required formatting. Adjust the user-defined parameters in the `src/eval` python scripts and run to produce confusion, sample and site level performance analyses. The results generated by these analyses can be visualized with the R scripts in `figures`.

For details regarding training or performance evaluation parameters and implementation, refer to the commented documentation header of the relevant script.

