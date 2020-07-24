# -*- coding: utf-8 -*-
"""
 Copyright (c) 2019-2020 Masahiko Hashimoto <hashimom@geeko.jp>

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
import csv
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cutiefake.train import Bert
from cutiefake.model import ELilyModel

BERT_EMB_DIM = 768


class Builder:
    def __init__(self, link_file, bert_path, output_dir):
        """ モデル生成

        :param link_file:
        :param output_dir:
        """
        self.link_list = []
        with open(link_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                self.link_list.append(row)

        # モデル配置ディレクトリ生成
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.out_dir = output_dir

        # モデル定義
        self.model = ELilyModel(BERT_EMB_DIM)
        self.use_device = "cpu"
        if torch.cuda.is_available():
            self.use_device = "cuda"
            self.model.to(self.use_device)

        # BERTモデル定義
        self.bert = Bert(bert_path, True)

    def __call__(self, epoch_num, batch_size):
        """ 学習実行

        :param epoch_num:
        :param batch_size:
        :return:
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 学習開始
        self.model.train()
        for i in range(epoch_num):
            # バッチリスト生成 ※ゆくゆくはデータセットへ移行する予定
            batch_list = []
            rand_list = np.random.randint(0, len(self.link_list), batch_size)
            for rand_i in rand_list:
                batch_list.append(self.link_list[rand_i])
            batch = self.make_batch(batch_list)

            # 学習データを入力して損失値を取得
            y = self.model(batch)
            loss = criterion(y, batch)

            # 勾配を初期化してBackProp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[%d] loss: %f" % (i, loss))

        torch.save(self.model.state_dict(), self.out_dir + "/dnn.mdl")

    def make_batch(self, batch_list):
        """　バッチ生成 ※将来的にデータセットへ移行する

        :param batch_list:
        :return:
        """
        batch = torch.empty([len(batch_list), BERT_EMB_DIM], device=self.use_device)
        for i, list_str in enumerate(batch_list):
            for j, sentence in enumerate(list_str):
                # 対象となる文節のみを取得。係り受け先は現状見ていない
                emb = self.bert.get_sentence_embedding(sentence, tokenize=True)
                batch[i] = emb.squeeze(0)
                break
        return batch


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-l', '--link_file', help='input link file', required=True)
    arg_parser.add_argument('-b', '--bert_model_path', help='bert model path', required=True)
    arg_parser.add_argument('-o', '--model_path', help='output model path', required=True)
    args = arg_parser.parse_args()

    builder = Builder(args.link_file, args.bert_model_path, args.model_path)
    builder(200, 256)


if __name__ == "__main__":
    main()



