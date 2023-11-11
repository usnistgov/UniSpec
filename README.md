# UniSpec
Implementation of UniSpec, a deep learning model for predicting full fragment ion peptide spectra.

This repository contains code for the implementation of UniSpec, from the publication:

UniSpec: A Deep Learning Approach for Predicting Energy-Sensitive Peptide Tandem Mass Spectra and Generating 
Proteomics-Wide In-Silico Spectral Libraries (2023).
https://www.biorxiv.org/content/10.1101/2023.06.14.544947v1

The code allows for the reconstruction of the entire project, from raw data to trained model to predicted libraries. 
Alternatively, you can use the trained model weights that are included in this repository (saved_models/unispec23).
The predict.yaml file paths are preset for this model.

## Contact Us ##

If you have any inquiries regarding the repository, please don't hesitate to reach out to the NIST UniSpec team. You can contact them directly at qian.dong@nist.gov, joellapin@comcast.net, and xinjian.yan@nist.gov. Alternatively, you can also post an issue in the repository itself.

The UniSpec team consists of associates and members of the Mass Spectrometry Data Center, Biomolecular Measurement Division of the Material Measurement Laboratory.

Installation
-
- git clone https://github.com/usnistgov/UniSpec

Raw data
--------
In order to turn raw mass spectral data from .msp files into streamlined datasets for model training, you must use
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

