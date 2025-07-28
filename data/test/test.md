This directory contains data for model performance evaluation with a held-out test dataset, including:

- A subdirectory `audio` containing all audio files for test evaluation
- A table `test_data_annotations.csv` containing all associated annotations for those files
- A matrix `site_presence_absence.csv` containing the ground truth site level data per site-class combination
- A table `site_metadata.csv` containing site-specific metadata

The `data/test/audio` directory should contain all audio files comprising the held-out test dataset. The `test_data_annotations.csv` table associates annotation labels with each individual test audio file, and uses the example format shown below, where "file" indicates the name of the audio file (including file extension), "labels" a comma-separated list of class labels associated with the file, and "focal" the original predicted focal class of that indicates the meaning of an "unknown" or "not_target" annotation (for example, the "unknown" label in row 3 below indicates that the file may or may not contain a Northern Saw-whet Owl vocalization; similarly, the "not_target" [i.e. not focal class] label in row 5 indicates that the file definitively does not contain a Northern Saw-whet Owl vocalization, but may contain a vocalization of one or more other species):

| focal_class | file    | labels  |
| -------- | ------- | ------- |
| Aegolius acadicus_Northern Saw-whet Owl  | Northern Saw-whet Owl-0.1218131317407667_SMA00380_20200501_071720.wav | "Corvus corax_Common Raven" |
| Aegolius acadicus_Northern Saw-whet Owl  | Northern Saw-whet Owl-0.1024243000084736_SMA00351_20200428_102356.wav | "Glaucidium californicum_Northern Pygmy-Owl, not_target" |
| Aegolius acadicus_Northern Saw-whet Owl  | Northern Saw-whet Owl-0.1273449130528034_SMA00556_20200627_030121.wav | "unknown" |
| Aegolius acadicus_Northern Saw-whet Owl  | Northern Saw-whet Owl-0.1340103467953624_SMA00380_20200501_071432.wav | "Setophaga nigrescens_Black-throated Gray Warbler, Corvus corax_Common Raven" |
| Aegolius acadicus_Northern Saw-whet Owl  | Northern Saw-whet Owl-0.3763227164745331_SMA00404_20200619_153545.wav | "not_target" |
| ...  | ... | ... |

The `site_presence_absence.csv` matrix contains the ground truth presence-absence data per site-class combination, where individual class observations (rows of 1s, 0s, and ?s) are associated with site-specific metadata (columns, e.g. site ID, ARU serial number, survey deployment number, etc.):

| site   | <site 1 ID>    | <site 2 ID> | ... |
| -------- | ------- | ------- | ------- |
| serialno | SMA00349 | SMA00309 | ... |
| deployment | <deployment #> | <deployment #> | ... |
| marbled murrelet | 1 | 0 | ... |
| rufous hummingbird | 0 | 1 | ... |
| brown creeper | 1 | ? | ... |
| western tanager | ? | 1 | ... |
| ...  | ... | ... | ... |

The `site_metadata.csv` table contains additional metadata relating to individual ARU site and survey deployments.