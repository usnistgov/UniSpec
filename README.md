# UniSpec
Implementation of UniSpec, a deep learning model for predicting full fragment ion peptide spectra.

This repository contains code for the implementation of UniSpec, from the publication:

UniSpec: A Deep Learning Approach for Predicting Energy-Sensitive Peptide Tandem Mass Spectra and Generating 
Proteomics-Wide In-Silico Spectral Libraries (2023).
https://www.biorxiv.org/content/10.1101/2023.06.14.544947v1

The code allows for the reconstruction of the entire project, from raw data to trained model to predicted libraries. 
Alternatively, if you have the checkpoint of a previously trained model, you can clone the repository directly
and run predictions immediately.

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

python Train.py 

The configuration settings for the model, data, and training procedure are set in the yaml file 
input_data/configuration/Train.yaml.

Predictions of evaluation
-------------------------
A trained model can be deployed for predictions by running the command:

python predict.py

The configuration settings for what/how you want to create predictions are set in the yaml file 
input_data/configuration/predict.yaml.
