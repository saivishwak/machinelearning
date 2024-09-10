# Llama2

This is an implementation from Umar Jamil (https://www.youtube.com/watch?v=oM4VmoabDAI&t=2072s), Supports both CPU and GPU inference.

### Papers

- https://arxiv.org/abs/2307.09288
- https://arxiv.org/abs/2302.13971
- [RotaryEmbeddings](https://arxiv.org/abs/2104.09864)

### Resources

- https://huggingface.co/docs/transformers/main/en/model_doc/llama2
- Download LlaMa weights from https://llama.meta.com/llama-downloads/

## Instructions

- Create virutal env and install the dependencies

- Post installation of deps run

```sh
pip install -e .
```

- Download the Llama2 weights and tokenzier files and move them to a folder 'weights' (need to create a new folder 'weights')

## For Text Completion

```sh
python3 inference.py
```
