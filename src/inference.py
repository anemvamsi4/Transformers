from src.model import lyricGPT
from src.utils import Config, BPETokenizer

import torch
import torch.functional as F
from tqdm import tqdm

# Inference Generator
def generate_sequences(model, tokenizer, new_tokens: int = 100 , prompt: str = " "):
    model.eval()
    device = next(model.parameters()).device

    prompt_tokens = tokenizer.encode(prompt)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
      for _ in tqdm(range(new_tokens)):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

        gen_idx = x[0].tolist()

    generated_sequence = tokenizer.decode([str(idx) for idx in gen_idx])

    return generated_sequence

if __name__ == '__main__':

    # Initialize tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load_vocab('./dataset/bpe_tokenizer')

    model_path = './model-logs/best_model.pt'

    model = lyricGPT(Config())
    model.load_state_dict(torch.load(model_path))

    #Generation after loading trained model:
    generated = generate_sequences(model, tokenizer, new_tokens = 100, prompt="నా పేరు ")
    print(f"\n\n{generated}")

    with open('generated.txt', 'w') as f:
        f.write(generated)