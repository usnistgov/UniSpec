# UniSpec
Implementation of UniSpec, a deep learning model for predicting full fragment ion peptide spectra.

This repository contains code for the implementation of UniSpec, from the publication:

UniSpec: Deep Learning for Predicting the Full Range of Peptide Fragment Ion Series to Enhance the Proteomics Data Analysis Workflow, Cite this: Anal. Chem. 2024, 96, 7, 2783–2790 https://pubs.acs.org/doi/10.1021/acs.analchem.3c02321

The code allows for the reconstruction of the entire project, from raw data to trained model to predicted libraries. 
Alternatively, you can use the trained model weights that are included in this repository (saved_models/unispec23).
The predict.yaml file paths are preset for this model.

UPDATE 10/6/2024:
Remote prediction platform service Koina now hosts a version of Unispec that outputs the top 200 predicted peaks by intensity. Please visit https://koina.wilhelmlab.org/ for
usage instructions.

## Contact Us ##

If you have any inquiries regarding the repository, please don't hesitate to reach out to the UniSpec team. You can contact us directly at qiand3161@gmail.com, xinjian.yan@nist.gov, and joellapin@comcast.net, Alternatively, you can also post an issue in the repository itself.

Installation
-
- git clone https://github.com/usnistgov/UniSpec

Raw data
--------
In order to turn raw mass spectral data from NIST MSP files into streamlined datasets for model training, you must use
the python script create_dataset_aidata.py in the scripts/ directory. The settings for this script are set in the
yaml files scripts/input_options/create_dataset.yaml and scripts/input_options/combo.yaml. Directions for using
the script create_dataset.py are in the comments at the top of that script.

6/21/2023
Caution: This script was written specifically for the data originally used in the publication, and is untested on 
other datasets.

Training a model
----------------
Training can be run with the command: 

- python Train.py 

The configuration settings for the model, data, and training procedure are set in the yaml file 
input_data/configuration/Train.yaml.

Predictions of evaluation
-------------------------
A trained model can be deployed for predictions by running the command:

- python predict.py

The configuration settings for what/how you want to create predictions are set in the yaml file 
input_data/configuration/predict.yaml.

Contributing
------------

At NIST, we provide NIST projects as a public service, and we highly value feedback and contributions from the community. If you have any contributions to make, we encourage you to actively participate in the development of this project. You can do so by forking this project, opening a pull request (PR), or initiating a discussion.

The authors of this project are eager to foster further innovations in the deep learning prediction of peptide fragmentation spectra, and your contributions can play a vital role in advancing this effort.

