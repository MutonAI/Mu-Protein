# Introduction 

The folder primarily hosts the code for pre-training pairwise masked language model for protein (PMLM). 

# Environment

Follow the steps below to set up the Conda environment:
```
conda create -n mutation python==3.8  
conda activate mutation  
pip install -r requirements.txt  
```

Additionally, you need to install PyTorch. The version to be installed is dependent on your GPU driver version. For instance:
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Or for a cpu-only version:
```
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

# Getting Started

## Data Preparation

Download and preprocess the protein sequence data into two text files: one for training and one for validation. Each sequence should:  

- Be **written in uppercase letters**  
- Have **spaces inserted between residues**  

For example, each line should be formatted as:  
```
C A S E F W S A W F ... C A D
```

After formatting, use `fairseq-preprocess` to convert these files into Fairseq binary format:  

```
fairseq-preprocess \
  --only-source \
  --trainpref <path_to_sequence_file>/uniref50.train.seqs \
  --validpref <path_to_sequence_file>/uniref50.valid.seqs \
  --destdir <path_to_output_folder>/generated/uniref50 \
  --workers 120 \
  --srcdict pmlm/src/protein/dict.txt
```

## Pretraining the Model
To pretrain a model using PMLM with the preprocessed files, run:
```
cd pmlm
bash script/pretrain.sh
```

## Pretrained Model

A pretrained model is publicly available on [figshare](https://doi.org/10.6084/m9.figshare.26892355).
