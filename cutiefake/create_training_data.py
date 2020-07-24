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
import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue
from transformers import BertTokenizer, BertModel


class Parser:
    def __init__(self, out_path, bert_model_path):
        """ 解析

        :param out_path: 出力パス (directory)
        :param bert_model_path: BERTモデルパス (directory)
        """
        self.out_path = out_path

        # BERT pre_trained model load
        self.bert_pre_trained_model = BertModel.from_pretrained(bert_model_path)
        self.bert_pre_trained_model.to("cuda").eval()

        # for Tokenizer
        vocab_file_path = bert_model_path + "/vocab.txt"
        self.bert_tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, do_basic_tokenize=False)

    def __call__(self, in_file):
        """

        :param in_file: 入力ファイル
        :return:
        """
        base, _ = os.path.splitext(os.path.basename(in_file))
        self.in_file = in_file
        self.out_file = self.out_path + "/" + base

        bodies = []
        link_info = []
        with open(self.in_file) as f:
            for line in f.readlines():
                line_split = line.split()

                # 文
                if line_split[0] == "#":
                    body = []
                    link_info_body = []
                    chunk = None

                # 文節
                elif line_split[0] == "*":
                    if chunk is not None:
                        body.append(chunk)
                    chunk = []

                    # 係り受け情報登録
                    link_info_body.append(int(line_split[1][:-1]))

                # 補足？
                elif line_split[0] == "+":
                    pass

                # 文末
                elif line_split[0] == "EOS":
                    body.append(chunk)
                    bodies.append(body)
                    link_info.append(link_info_body)

                # 単語
                else:
                    chunk.append(line_split[0])

        # BERT Token 取得
        out_token = []
        # emb_zeros = np.zeros(768)
        for body_idx, body in enumerate(bodies):
            bert_tokens = []
            for chunk in body:
                tokens = self.bert_tokenizer.tokenize(" ".join(chunk))
                ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
                tokens_tensor = torch.tensor(ids).unsqueeze(0)
                bert_tokens.append(tokens_tensor)

            for phase_idx, tokens in enumerate(bert_tokens):
                link_tokens = []
                for link in link_info[body_idx]:
                    if phase_idx == link:
                        link_tokens.append([tokens, bert_tokens[link]])

                if len(link_tokens) != 0:
                    out_token.extend(link_tokens)
                else:
                    out_token.append([tokens, None])

        # to BERT emb
        out_emb = []
        for token in out_token:
            emb, _ = self.bert_pre_trained_model(token[0].to("cuda"))
            words_emb = torch.sum(emb[0].T, 1).detach().to("cpu").numpy()
            if token[1] is not None:
                emb, _ = self.bert_pre_trained_model(token[1].to("cuda"))
                link_emb = torch.sum(emb[0].T, 1).detach().to("cpu").numpy()
                out_emb.append([words_emb, link_emb])
            else:
                out_emb.append([words_emb, np.zeros(768, dtype=np.float32)])

        # save
        with open(self.out_file + ".pkl", "wb") as f:
            pickle.dump(out_emb, f)


def worker(in_q, out_q, out_path, bert_model_path):
    """ Worker

    :param in_q: キュー
    :param out_q: キュー
    :param out_path: 出力パス (directory)
    :param bert_model_path: BERTモデルパス (directory)
    :return:
    """
    p = Parser(out_path, bert_model_path)
    while not in_q.empty():
        file = in_q.get()
        p(file)
        out_q.put(file)


def main():
    """ main

    :return:
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_path', help='KWDLC path', required=True)
    arg_parser.add_argument('-t', '--in_type', help='Input Type ("train" or "test" or "all")', default="all")
    arg_parser.add_argument('-o', '--out_path', help='output path', default="out/")
    arg_parser.add_argument('-p', '--proc_num', help='process num', default=2)
    arg_parser.add_argument('-b', '--bert_model_path', help='BERT MODEL path', required=True)
    args = arg_parser.parse_args()

    # プロセス数
    proc_num = int(args.proc_num)

    out_path = os.path.abspath(args.out_path)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # リスト作成
    in_file_list = []
    if args.in_type != "test":
        # 学習用ファイルリスト追加
        with open(os.path.abspath(args.in_path + "/train.files")) as f:
            for file in f.readlines():
                in_file_list.append(file.rstrip('\n'))
    if args.in_type != "train":
        # テスト用ファイルリスト追加
        with open(os.path.abspath(args.in_path + "/test.files")) as f:
            for file in f.readlines():
                in_file_list.append(file.rstrip('\n'))

    # 入力リストの作成
    in_q = Queue()
    for path in in_file_list:
        in_q.put(os.path.abspath(args.in_path + "/" + path))

    proc = []
    out_q = Queue()
    for i in range(proc_num):
        p = Process(target=worker, args=(in_q, out_q, out_path, args.bert_model_path))
        p.start()
        proc.append(p)

    for _ in tqdm(range(len(in_file_list))):
        out_q.get()

    for p in proc:
        p.join()


if __name__ == "__main__":
    main()

