# -*- coding: utf-8 -*-
"""
 Copyright (c) 2020 Masahiko Hashimoto <hashimom@geeko.jp>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
"""
import argparse
import glob
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm import tqdm
from cutiefake.model import ELilyBert, BertEncoder, BertDecoder, ELilyModel
from torch.utils.data import Dataset


class BertDataSets(Dataset):
    def __init__(self, word_emb_file, transform=None):
        self.word_emb_file = word_emb_file
        self.transform = transform
        self.word_phase = []
        self.word_read = []
        self.words_emb = []

        with open(word_emb_file, 'rb') as f:
            while True:
                try:
                    pkl_data = pickle.load(f)
                    self.word_phase.append(pkl_data[0])
                    self.word_read.append(pkl_data[1])
                    self.words_emb.append(pkl_data[2])
                except EOFError:
                    # ファイル末尾のため終了
                    print("Word pickle file read end")
                    break

    def __len__(self):
        return len(self.words_emb)

    def __getitem__(self, idx):
        return self.word_phase[idx], self.word_read[idx], self.words_emb[idx]


class ELilyDataSets(Dataset):
    def __init__(self, link_files_dir, transform=None):
        self.transform = transform
        self.words_emb = []
        self.link_emb = []

        file_list = glob.glob(link_files_dir + "/*.pkl")
        for file in file_list:
            with open(file, 'rb') as f:
                pkl_data = pickle.load(f)
                for data in pkl_data:
                    self.words_emb.append(data[0])
                    self.link_emb.append(data[1].astype(np.float32))

    def __len__(self):
        return len(self.words_emb)

    def __getitem__(self, idx):
        return self.words_emb[idx], self.link_emb[idx]


class BertTrainer:
    def __init__(self, word_emb_file, output_dir):
        """ モデル生成

        :param word_emb_file:
        :param output_dir:
        """
        # データセット作成
        self.word_emb_file = word_emb_file
        self.dataset = BertDataSets(word_emb_file)

        # モデル配置ディレクトリ生成
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.out_dir = output_dir

        # モデル定義
        self.encoder = BertEncoder()
        self.decoder = BertDecoder()
        self.bert_model = ELilyBert(self.encoder, self.decoder)
        self.use_device = "cpu"
        if torch.cuda.is_available():
            self.use_device = "cuda"
            self.bert_model.to(self.use_device)

    def __call__(self, epoch_num, batch_size):
        """ 学習実行

        :param epoch_num:
        :param batch_size:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        # 学習開始
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.bert_model.parameters(), lr=0.001)
        self.bert_model.train()
        for i in range(epoch_num):
            batch_loss = 0.
            for _, _, word_emb in train_loader:
                x_in = word_emb.to(self.use_device)
                y = self.bert_model(x_in)
                loss = criterion(y, x_in)

                # 勾配を初期化してBackProp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss

            print("[BERT %d] loss: %f" % (i + 1, batch_loss / len(self.dataset)))

        torch.save(self.encoder.state_dict(), self.out_dir + "/encoder.mdl")
        torch.save(self.decoder.state_dict(), self.out_dir + "/decoder.mdl")

        print("words.csv output start")
        with open(self.word_emb_file, 'rb') as in_f:
            with open('words.csv', 'w') as out_f:
                writer = csv.writer(out_f)
                while True:
                    try:
                        pkl_data = pickle.load(in_f)
                        x_in = torch.from_numpy(pkl_data[2]).to(self.use_device)
                        y = self.encoder(x_in)
                        writer.writerow([pkl_data[0], pkl_data[1], y.to("cpu").item()])
                    except EOFError:
                        # ファイル末尾のため終了
                        print("Words csv output end")
                        break


class ELilyTrainer:
    def __init__(self, link_files_dir, output_dir):
        """ モデル生成

        :param link_files_dir:
        :param output_dir:
        """
        # データセット作成
        self.dataset = ELilyDataSets(link_files_dir)

        # モデル配置ディレクトリ生成
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.out_dir = output_dir

        # モデル定義
        self.elily_model = ELilyModel()
        self.use_device = "cpu"
        if torch.cuda.is_available():
            self.use_device = "cuda"
            self.elily_model.to(self.use_device)

    def __call__(self, epoch_num, batch_size):
        """ 学習実行

        :param epoch_num:
        :param batch_size:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        # 学習開始
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.elily_model.parameters(), lr=0.001)
        self.elily_model.train()
        for i in range(epoch_num):
            batch_loss = 0.
            for (phase, link) in train_loader:
                x_in = torch.cat([phase, link], dim=1)
                x_in = x_in.to(self.use_device)
                y = self.elily_model(x_in)
                loss = criterion(y, x_in)

                # 勾配を初期化してBackProp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss

            print("[ELily %d] loss: %f" % (i + 1, batch_loss / len(self.dataset)))

        torch.save(self.elily_model.state_dict(), self.out_dir + "/dnn.mdl")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-w', '--words_emb_file', help='word_emb file', required=True)
    arg_parser.add_argument('-l', '--linked_file', help='link file', required=True)
    arg_parser.add_argument('-o', '--output_path', help='output model path', required=True)
    args = arg_parser.parse_args()

    bert = BertTrainer(args.words_emb_file, args.output_path)
    bert(100, 2056)

    elily = ELilyTrainer(args.linked_file, args.output_path)
    elily(200, 1024)


if __name__ == "__main__":
    main()
