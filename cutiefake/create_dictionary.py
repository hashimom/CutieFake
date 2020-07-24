# -*- coding: utf-8 -*-
"""
 Copyright (c) 2018-2019 Masahiko Hashimoto <hashimom@geeko.jp>

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
import os
import csv
import argparse
import glob
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from cutiefake.model import BertEncoder


class MecabDicReader:
    def __init__(self, in_csv_dir, out_csv_dir, elily_model_path, bert_model_path):
        self.in_csv_dir = os.path.abspath(in_csv_dir)
        self.out_csv_dir = os.path.abspath(out_csv_dir + "/")

        # BERT pre_trained model load
        self.bert_pre_trained_model = BertModel.from_pretrained(bert_model_path)
        self.bert_pre_trained_model.to("cuda").eval()

        # for Tokenizer
        vocab_file_path = bert_model_path + "/vocab.txt"
        self.bert_tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, do_basic_tokenize=False)

        self.encoder = BertEncoder()
        self.encoder.load_state_dict(torch.load(elily_model_path + "/encoder.mdl"))
        self.encoder.to("cuda").eval()

    def __call__(self):
        word_list_from_dic = []

        print("Mecab dictionary reading...")
        csv_list = glob.glob(self.in_csv_dir + "/*.csv")
        for csv_file in csv_list:
            print(csv_file + " open")
            with open(csv_file, "r") as f:
                reader = csv.reader(f, delimiter=",")
                for row in reader:
                    word_list_from_dic.append([row[0], row[9]])

        with open("word_emb.pkl", "wb") as f:
            for word in tqdm(word_list_from_dic):
                try:
                    bert_tokens = self.bert_tokenizer.tokenize(word[0])
                    ids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
                    tokens_tensor = torch.tensor(ids).unsqueeze(0).to("cuda")
                    emb, _ = self.bert_pre_trained_model(tokens_tensor)
                    emb = torch.sum(emb[0].T, 1).detach().to("cpu").numpy()
                    pickle.dump([word[0], word[1], emb], f)
                except Exception as e:
                    print(e)
                    print("[err] %s / %s" % (word[0], word[1]))
                    continue


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_path', help='link file', required=True)
    arg_parser.add_argument('-o', '--out_path', help='output path', required=True)
    arg_parser.add_argument('-m', '--model_path', help='model path', required=True)
    arg_parser.add_argument('-b', '--bert_path', help='model path', required=True)
    args = arg_parser.parse_args()

    reader = MecabDicReader(args.in_path, args.out_path, args.model_path, args.bert_path)
    reader()


if __name__ == "__main__":
    main()


