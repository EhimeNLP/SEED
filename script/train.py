import argparse
import time
import sys
import torch
import torch.nn as nn
import random
from model import MLP_large
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

parser = argparse.ArgumentParser()
# PATH
parser.add_argument("--save_model_path", default="../output/1M_200k_50k/SEED")
# train data path
parser.add_argument("--train_path", default="../data/train_wmt20/train_1M_200k_50k.pt")
# valid data path
parser.add_argument("--valid_path", default="../data/train_wmt20/valid_1M_200k_50k.pt")
# train parameter
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lang_num", type=int, default=7)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=100)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num2lang = {0: "en", 1: "de", 2: "zh", 3: "ro", 4: "et", 5: "ne", 6: "si"}

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, src_emb, trg_emb, src_lang, trg_lang):
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __len__(self):
        return len(self.src_emb)

    def __getitem__(self, idx):
        return {
            "src_emb": self.src_emb[idx],
            "trg_emb": self.trg_emb[idx],
            "src_lang": self.src_lang[idx],
            "trg_lang": self.trg_lang[idx],
        }
        
def rand_sentence(embs, langs):
    batch_size = embs.size(0)
    rand_sentence = []

    for i in range(batch_size):
        lang = langs[i].int().item()
        same_lang_idxs = (langs == lang).nonzero(as_tuple=True)[0]
        
        same_lang_idxs = same_lang_idxs[same_lang_idxs != i]
        
        rand_idx = random.choice(same_lang_idxs.tolist()) if len(same_lang_idxs) > 0 else i 
        rand_sentence.append(embs[rand_idx])

    # テンソルに変換して返す
    rand_sentence = torch.stack(rand_sentence)
    return rand_sentence


def calculate_loss(src_emb_i, trg_emb_i, la_src_i, me_src_i, la_src_j, me_src_j, la_trg_i, me_trg_i, la_trg_j, me_trg_j, writer, mode, epoch):
    cos_fn = nn.CosineEmbeddingLoss()
    y = torch.ones(la_src_i.size(0), device=device)
    #分離することの学習
    #(原｜目的)言語の言語表現 ←→　(原｜目的)言語の意味表現
    loss_emb_sep = cos_fn(la_src_i, me_src_i, -y) + cos_fn(la_trg_i, me_trg_i, -y)
    
    # 意味の対照学習
    loss_me = cos_fn(me_src_i, me_trg_i, y) * 2 + cos_fn(me_src_i, me_src_j, -y) + cos_fn(me_trg_i, me_trg_j, -y)
    
    # 交差復元の対照学習
    loss_cross = cos_fn(src_emb_i, la_src_i + me_trg_i, y) + cos_fn(trg_emb_i, la_trg_i + me_src_i, y) + cos_fn(src_emb_i, la_src_j + me_src_i, y) + cos_fn(trg_emb_i, la_trg_j + me_trg_i, y)
    
    # 言語の学習
    loss_la_all = cos_fn(la_src_i, la_src_j, y) + cos_fn(la_trg_i, la_trg_j, y)
    
    total_loss = loss_emb_sep + loss_me + loss_cross + loss_la_all
    
    losses = {'total_loss': total_loss,
              'loss_emb_sep': loss_emb_sep,
              'loss_me': loss_me,
              'loss_cross': loss_cross,
              'loss_la_all': loss_la_all,
              }
    
    return losses
    

def train_model(
    model,
    dataset_train,
    dataset_valid,
    optimizer,
    batch_size,
    save_model_path,
):
    writer = SummaryWriter(log_dir= save_model_path + "/log/")
    model.to(device)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    min_valid_loss = float("inf")
    for epoch in range(10000):
        # train
        s_time = time.time()
        train_loss = 0
        print(epoch)
        cnt_step = 0
        mode = "train"
        loss_emb_sep_train = 0
        loss_me_train = 0
        loss_cross_train = 0
        loss_la_all_train = 0

        loss_emb_sep_eval = 0
        loss_me_eval = 0
        loss_cross_eval = 0
        loss_la_all_eval = 0

        for data in dataloader_train:
            src_emb_i = data["src_emb"].to(device)
            src_lang_i = data["src_lang"].to(device)
            
            trg_emb_i = data["trg_emb"].to(device)
            trg_lang_i = data["trg_lang"].to(device)
            
            src_emb_j = rand_sentence(src_emb_i, src_lang_i).to(device)
            trg_emb_j = rand_sentence(trg_emb_i, trg_lang_i).to(device)
            
            #iの処理
            me_src_i = model(src_emb_i)
            la_src_i = (src_emb_i - me_src_i).to(device)
            me_trg_i = model(trg_emb_i)
            la_trg_i = (trg_emb_i - me_trg_i).to(device)
            
            #jの処理
            me_src_j = model(src_emb_j)
            la_src_j = (src_emb_j - me_src_j).to(device)
            me_trg_j = model(trg_emb_j)
            la_trg_j = (trg_emb_j - me_trg_j).to(device)

            optimizer.zero_grad()
            losses  = calculate_loss(src_emb_i, trg_emb_i, la_src_i, me_src_i, la_src_j, me_src_j, la_trg_i, me_trg_i, la_trg_j, me_trg_j, writer, epoch, mode)
            train_loss += losses['total_loss'].item()
            loss_emb_sep_train += losses['loss_emb_sep'].item()
            loss_me_train += losses['loss_me'].item()
            loss_cross_train += losses['loss_cross'].item()
            loss_la_all_train += losses['loss_la_all'].item()
            losses['total_loss'].backward()
            optimizer.step()
            cnt_step += 1
            print("train loss calculated..." + str(cnt_step), flush=True)
        
        # 全部外に
        writer.add_scalar(mode + "/train_loss", train_loss, epoch)
        writer.add_scalar(mode + "/loss_emb_sep", loss_emb_sep_train, epoch)
        writer.add_scalar(mode + "/loss_me", loss_me_train, epoch)
        writer.add_scalar(mode + "/loss_cross", loss_cross_train, epoch)
        writer.add_scalar(mode + "/loss_la_all", loss_la_all_train, epoch)

        # eval
        with torch.no_grad():
            valid_loss = 0
            mode = "valid"
            print("検証データで実行中:epoch" + str(epoch), flush=True)
            for data in dataloader_valid:
                src_emb_i = data["src_emb"].to(device)
                src_lang_i = data["src_lang"].to(device)
                
                trg_emb_i = data["trg_emb"].to(device)
                trg_lang_i = data["trg_lang"].to(device)
                
                src_emb_j = rand_sentence(src_emb_i, src_lang_i).to(device)
                trg_emb_j = rand_sentence(trg_emb_i, trg_lang_i).to(device)

                #iの処理
                me_src_i = model(src_emb_i)
                la_src_i = (src_emb_i - me_src_i).to(device)
                me_trg_i = model(trg_emb_i)
                la_trg_i = (trg_emb_i - me_trg_i).to(device)
                
                #jの処理
                me_src_j = model(src_emb_j)
                la_src_j = (src_emb_j - me_src_j).to(device)
                me_trg_j = model(trg_emb_j)
                la_trg_j = (trg_emb_j - me_trg_j).to(device)
                
                losses = calculate_loss(
                    src_emb_i, trg_emb_i, la_src_i, me_src_i, la_src_j, me_src_j, la_trg_i, me_trg_i, la_trg_j, me_trg_j, writer, epoch, mode
                )
                valid_loss += losses['total_loss'].item()
                loss_emb_sep_eval += losses['loss_emb_sep'].item()
                loss_me_eval += losses['loss_me'].item()
                loss_cross_eval += losses['loss_cross'].item()
                loss_la_all_eval += losses['loss_la_all'].item()
                
            writer.add_scalar(mode + "/valid_loss", valid_loss, epoch)
            writer.add_scalar(mode + "/loss_emb_sep", loss_emb_sep_eval, epoch)
            writer.add_scalar(mode + "/loss_me", loss_me_eval, epoch)
            writer.add_scalar(mode + "/loss_cross", loss_cross_eval, epoch)
            writer.add_scalar(mode + "/loss_la_all", loss_la_all_eval, epoch)

            print(
                f"epoch:{epoch + 1: <2}, "
                f"train_loss: {train_loss / len(dataloader_train):.5f}, "
                f"valid_loss: {valid_loss / len(dataloader_valid):.5f}, "
                f"{(time.time() - s_time) / 60:.1f}[min]",
                flush=True
            )

            if valid_loss < min_valid_loss:
                epochs_no_improve = 0
                min_valid_loss = valid_loss
                torch.save(model.to("cpu").state_dict(), save_model_path + "/log/" + "/best_val.pt")
                model.to(device)
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= 5:
            break


def main():
    log_dir = os.path.join(args.save_model_path, "log")
    
    # ディレクトリが存在しない場合に作成
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイル名を指定
    log_file_name = os.path.join(log_dir, "train.log.txt")
    
    with open(log_file_name, "w") as log_file:
        sys.stdout = log_file  # printの出力先をファイルに変更
        
        # ログファイルに学習パラメータを記録
        print(f"Learning Rate: {args.lr}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Training Path: {args.train_path}")
        print(f"Validation Path: {args.valid_path}")
    
        data_train = torch.load(args.train_path)
        dataset_train = TextDataset(
            data_train["src_emb"], data_train["trg_emb"], data_train["src_lang"], data_train["trg_lang"]
        )
        data_valid = torch.load(args.valid_path)
        dataset_valid = TextDataset(
            data_valid["src_emb"], data_valid["trg_emb"], data_valid["src_lang"], data_valid["trg_lang"]
        )

        model = MLP_large().cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_model(
            model,
            dataset_train,
            dataset_valid,
            optimizer,
            args.batch_size,
            args.save_model_path,
        )


if __name__ == "__main__":
    main()