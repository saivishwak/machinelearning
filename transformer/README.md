Vanilla Transformer Model for translation task

Paper -
https://arxiv.org/abs/1706.03762

Dataset -
https://ai4bharat.iitm.ac.in//samanantar/

English - Hindi Translation
BPE Tokenizer

Original Dataset
Max length of English sentence: 16432
Max length of Hindi sentence: 8609

Adjusted the data by removing the top biggest sentences so that the model can fit on a single RTX4090 GPU (A cap of 1000 tokens is also added in code)
Max length of English sentence post adjust: 3822
Max length of Hindi sentence post adjust: 2294

The model is taking long to train on a single RT4090 GPU. Will be researching on how can we make it train with less params or check for cloud GPU.

### Instructions

- Install all the python dependencies
- Download the dataset and copy it to data folder (need to create the folder)
- Create a weights folder

To run training

```sh
python3 train.py
```
