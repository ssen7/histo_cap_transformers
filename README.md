# histo_cap_transformers

Paper Link: https://arxiv.org/abs/2312.01435. Under consideration at ISBI 2024.

This work builds on my prior work on
[HistoCap](https://github.com/ssen7/histo_cap_generation_2), accepted in ML4H 2023 Findings track, by incorporating BERT based decoder instead of a LSTM decoder. We are also able to include tissue type, gender and the actual caption into the actual caption generated, while we were limited by vocabulary in case of an LSTM based decoder.

Here is the [inference pipeline](https://github.com/ssen7/histo_caption_inference_pipeline).

Abstract:

Deep learning for histopathology has been successfully used for disease classification, image segmentation and more. However, combining image and text modalities using current state-of-the-art methods has been a challenge due to the high resolution of histopathology images. Automatic report generation for histopathology images is one such challenge. In this work, we show that using an existing pre-trained Vision Transformer in a two-step process of first using it to encode 4096x4096 sized patches of the Whole Slide Image (WSI) and then using it as the encoder and a pre-trained Bidirectional Encoder Representations from Transformers (BERT) model for language modeling-based decoder for report generation, we can build a fairly performant and portable report generation mechanism that takes into account the whole of the high resolution image, instead of just the patches. Our method allows us to not only generate and evaluate captions that describe the image, but also helps us classify the image into tissue types and the gender of the patient as well. Our best performing model achieves a 79.98% accuracy in Tissue Type classification and 66.36% accuracy in classifying the sex of the patient the tissue came from, with a BLEU-4 score of 0.5818 in our caption generation task.


To run the lightning scripts, use requirements.txt to install the necessary packages and then run the training scripts. Details on the different script are given bellow.

- data_files/ directory: Contains train/test/val pandas dataframe stored as a pickle file to preserve data types.
- model_vit_bert.py: Contains model definitions
- dataloader.py: Contains the custom PyTorch dataloader.
- patch_4k_h5.py: Contains the code for patching high resolution WSIs in the form SVS images and save the patches in an hdf5 format.
- generate4k_256clsreps.py: Extract the representations from pre-trained ViT. WARNING: To run this script download the following GitHub repo: https://github.com/mahmoodlab/HIPT


### Training scripts:

- Train script: training_script_vit_bert_5layers.py.
- Evaluation script: evaluation5layers.py
