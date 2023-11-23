# histo_cap_transformers
To run the lightning scripts, use requirements.txt to install the necessary packages and then run the training scripts. Details on the different script are given bellow.

- data_files/ directory: Contains train/test/val pandas dataframe stored as a pickle file to preserve data types.
- model_vit_bert.py: Contains model definitions
- dataloader.py: Contains the custom PyTorch dataloader.
- patch_4k_h5.py: Contains the code for patching high resolution WSIs in the form SVS images and save the patches in an hdf5 format.
- generate4k_256clsreps.py: Extract the representations from pre-trained ViT. WARNING: To run this script download the following GitHub repo: https://github.com/mahmoodlab/HIPT


### Training scripts:

- Train script: training_script_vit_bert_5layers.py.
- Evaluation script: evaluation5layers.py
