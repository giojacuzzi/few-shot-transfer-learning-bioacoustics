# few-shot-transfer-learning-bioacoustics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15694435.svg)](https://doi.org/10.5281/zenodo.15694435)

This repository contains the open source software, workflow routines, and reproducible methods presented in the research article:

> Jacuzzi, G., Olden, J.D., 2025. Few-shot transfer learning enables robust acoustic monitoring of wildlife communities at the landscape scale. Ecological Informatics 90, 103294. [doi.org/10.1016/j.ecoinf.2025.103294](https://doi.org/10.1016/j.ecoinf.2025.103294)

Although this software was initially developed for a particular study region and set of monitoring objectives (avian biodiversity surveys in the Olympic Experimental State Forest of Washington, USA), it is designed to be freely repurposed and we encourage its use in other applications. Please cite the original publication in your references. Direct any correspondance to gioj@uw.edu, and request features or bug fixes via GitHub issues.

### Recommendations
- Visual Studio Code with Microsoft extensions [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy), and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
- Install dependencies to a [virtual python environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (.venv) that is used exclusively for this project. Create and activate the environment manually, or via VS Code.
- Download example data via [the latest release assets on GitHub](https://github.com/giojacuzzi/few-shot-transfer-learning-bioacoustics/releases) to reproduce results from the research article.

### Prerequisites
- Windows 7+ or MacOS 10.12+ operating system (64-bit). Note that the version of TensorFlow currently used alongside BirdNET is not natively compatible with Apple Silicon.
- [Python](https://www.python.org/downloads/) 3.9+ 64-bit (3.10 recommended, ensure "Add path to environment variables" is checked during install).

### Package dependencies
From a terminal shell within the virtual environment, navigate to the root directory of this repository (`few-shot-transfer-learning-bioacoustics`), and run:

```
git submodule update --init --recursive
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required dependencies, including a custom fork of BirdNET (version 2.4) with additional tools for model training and development. For troubleshooting reference, see setup instructions for [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer).

## Contents
- `data` – data for model training and testing
- `models` - files for source, target, and ensemble models, including associated class label lists 
- `results` - results (performance metrics, figures) generated from `src` scripts
- `src` – all source code for data annotation, processing, and analysis
    - `figures` – R scripts to generate figure plots and data visualizations
    - `gui` – supporting files for the GUI application
    - `misc` – helper modules and logging
    - `perf` – functions for calculating raw sample and site level performance metrics
    - `submodules` – repository dependencies (a fork of BirdNET-Analyzer with customized model training routines)

## Audio classification with BirdNET and/or custom model

A command-line interface script is provided to process audio data with a custom and/or pre-trained classifier. Run `src/process_audio.py` to produce predictions for a given directory or file with the specified model(s). Show arguments with `python src/process_audio.py -h`, or see documentation within the script itself.

#### GUI application

Alternatively, audio can be processed via a graphical user interface application with `src/gui.py`. To build and package the GUI as an executable from source, first activate the virtual environment from the console, then run `pyinstaller gui.spec --clean`.

> *Quickstart:* Rather than build the GUI application from source, precompiled executables for MacOS and Windows are provided via [the latest GitHub release](https://github.com/giojacuzzi/few-shot-transfer-learning-bioacoustics/releases). Download and unzip the corresponding file for your platform, then launch the `Model Ensemble Interface` executable.
>
> With the application window open, configure an analysis and/or segment process by dragging-and-dropping files or directories and adjusting the interface options. A configuration can be saved as a `.json` *session* file (`File > Save session`), and opened for repeated use (`File > Open session`). Click the `Launch process` button to begin your analysis, and monitor progress via the console log.

## Model training and performance evaluation pipeline

> *Quickstart:* To begin with a working example of the pipeline, download sample training and test data included with [the latest GitHub release](https://github.com/giojacuzzi/few-shot-transfer-learning-bioacoustics/releases). For a detailed description of the pipeline and these data consult [Jacuzzi and Olden 2025](https://doi.org/10.1016/j.ecoinf.2025.103294) and the script documentation.

### Training via few-shot transfer learning

To train a custom model, first assemble a `data/training` directory containing all audio data files for development (including training and validation) and a table containing all associated data annotations. See `data/training/training.md` for details and required formatting.

To prepare the development data for model training see `src/submodules/BirdNET-Analyzer/pretrain_stratified_kfold.py`. By default, development data are split into sets for model training and k-fold cross-validation with iterative stratification (see script documentation). Adjust the user-defined arguments of the script if desired, then run to generate a table of development file references for training and validation datasets. The script will output the required command(s) to conduct training for your requested model(s), following the format:

```
cd src/submodules/BirdNET-Analyzer
python3 train_fewshot.py --i /data/cache/<model>/training/<model>_I<split>/combined_development_files.csv --l /models/target/target_species_list.txt --o /models/target/<model>/<model>_I<split>/<model>_I<split>.tflite --no-autotune --epochs <epochs> --learning_rate <learning_rate> --batch_size <batch_size> --hidden_units <hidden_units>
```

Executing these commands will launch the model training process (see details in `src/submodules/BirdNET-Analyzer/train_fewshot.py`). After the training process is complete, follow the next section to conduct a thorough model performance evaluation of your model(s).

### Model performance evaluation
To evaluate model performance, first assemble a `data/test` directory containing all evaluation (i.e. validation or test data) audio files and a table containing all evaluation data annotations. See `data/test/test.md` for required formatting.

The following scripts for performance evaluation are provided in the `src` directory:

| Script                        | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `eval_segment_perf.py`        | Quantify audio segment level performance (e.g. PR AUC) of a custom target model on an evaluation (validation or test) dataset, and compare performance with a pre-trained source model. |
| `eval_confusion.py`           | Summarize patterns of confusion between classes by quantifying the frequency of occurrence in incorrectly classified species predictions (false positives and false negatives). |
| `eval_site_perf.py`           | Evaluate site level (i.e. presence-absence, species richness) performance metrics for target and source models with an evaluation dataset. |
| `eval_site_perf_by_factor.py` | Evaluate site level performance metrics across categorical factors (e.g. habitat types) for target and source models, including all classes and only shared classes. |

Note that these scripts utilize caching of intermediate data under `data/cache` to considerably reduce the time required for subsequent runs. The results generated by these analyses are written to the `results` directory and can be visualized with the R scripts in `src/figures`. For details regarding performance evaluation implementation and user-defined arguments available for adjustment, refer to the documentation header of the relevant script.

