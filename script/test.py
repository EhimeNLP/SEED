import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
from model import MLP_large

parser = argparse.ArgumentParser()
parser.add_argument("--wmt20", action="store_true")
parser.add_argument("--wmt21", action="store_true")
parser.add_argument("--model_path", default="../output/1M_200k_50k/best_val_custom.pt")
parser.add_argument("--batch_size", default=512)
parser.add_argument("--dream", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = "intfloat/multilingual-e5-large-instruct"

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embedding(tokenizer, base_model, model, sentences, batch_size, device):
    base_model.to(device)
    model.to(device)

    all_embeddings = []

    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    for i in range(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]

        encoded = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = base_model(**encoded.to(device))

        # last hidden state
        # embeddings = outputs[0][:, 0, :]
        embeddings = average_pool(outputs.last_hidden_state, encoded['attention_mask'])
        
        embeddings = model(embeddings)
        
        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = torch.stack(all_embeddings)

    return all_embeddings


def main():
    # if args.wmt20:
    #     print("Task: WMT20 QE")
    #     year = "test20"

    # if args.wmt21:
    #     print("Task: WMT21 QE")
    #     year = "test21"
    
    year = "test20"

    base_dir_path = f"../data/test/{year}"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    base_model.to(device)

    print(args.model_path)
    model = MLP_large()
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.to(device)

    lang_pairs = ["ende", "enzh", "roen", "eten", "neen", "sien"]

    pearson_scores = []  # Pearsonスコアを格納するリストを追加

    for lang_pair in lang_pairs:
        dir_path = os.path.join(base_dir_path, f"{lang_pair[:2] + '-' + lang_pair[2:]}-{year}")

        with open(os.path.join(dir_path, f"{year}.src"), "r") as f:
            src_sentences = f.read().rstrip()
            src_sentences = src_sentences.split("\n")

        with open(os.path.join(dir_path, f"{year}.mt"), "r") as f:
            trg_sentences = f.read().rstrip()
            trg_sentences = trg_sentences.split("\n")

        with open(os.path.join(dir_path, f"{year}.da"), "r") as f:
            da_scores = f.read().rstrip()
            da_scores = list(map(float, da_scores.split("\n")))
            da_scores = np.array(da_scores)

        src_emb = embedding(tokenizer, base_model, model, src_sentences, args.batch_size, device)
        trg_emb = embedding(tokenizer, base_model, model, trg_sentences, args.batch_size, device)
        
        cos = nn.CosineSimilarity()
        pred_score = cos(src_emb, trg_emb)

        pearson = pearsonr(pred_score.to("cpu").detach().numpy(), da_scores)[0]
        pearson = round(pearson, 3)
        print(f"{lang_pair}\n{pearson:.3f}")
        
        pearson_scores.append(pearson)  # スコアをリストに追加

    # Pearsonスコアの平均を計算して出力
    Avg = np.mean(pearson_scores)
    print(f"Avg\n{Avg:.4f}")


if __name__ == "__main__":
    main()
