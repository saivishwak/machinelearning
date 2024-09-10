import math
import torch

from llama2.generation import LlaMa

if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = LlaMa.build(checkpoints_dir="./weights/llama-2-7b/", tokenizer_path="./weights/llama-2-7b/tokenizer.model",
                        load_model=True, max_batch_size=3, max_seq_len=1024, device=device)

    print("All Ok")

    prompts = [
        "Translate Hello to Hindi. Output: ",
    ]
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
