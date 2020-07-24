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
import os
import csv
import argparse
import marisa_trie
import numpy as np
import torch
import torch.nn as nn
from cutiefake.words import Words
from cutiefake.model import BertDecoder, ELilyModel


BERT_EMB_DIM = 768
COST_TMP_MAX = 99  # dummy
ID_TMP_MAX = -1  # dummy


class Converter:
    def __init__(self, model_path):
        """ 変換モジュール

        :param model_path:
        """
        self.words = []
        trie_keys = []
        trie_values = []
        self.loss = nn.MSELoss()

        model_path = os.path.abspath(model_path)
        with open(model_path + "/words.csv", "r") as f:
            reader = csv.reader(f, delimiter=",", doublequote=True, quotechar='"')
            for i, row in enumerate(reader):
                # TrieのKeyには読み、Valueには単語ID（リストインデックス）を格納
                trie_keys.append(row[1])
                trie_values.append([i])
                # 単語IDに対応したデータをリストへ格納
                self.words.append({"w": row[0], "e": np.array([float(row[2])], dtype=np.float32)})

        self.trie = marisa_trie.RecordTrie("<I", zip(trie_keys, trie_values))
        self.word_info = Words()

        self.b_model = BertDecoder().to("cuda")
        self.b_model.load_state_dict(torch.load(model_path + "/decoder.mdl"))
        self.b_model.eval()

        self.e_model = ELilyModel().to("cuda")
        self.e_model.load_state_dict(torch.load(model_path + "/dnn.mdl"))
        self.e_model.eval()
        self.criterion = nn.MSELoss()

    def __call__(self, in_text):
        """ 変換

        :param in_text:
        :return:
        """
        # 入力が一文字の場合は単漢字変換へ移行 ※未実装
        in_len = len(in_text)
        if in_len <= 1:
            return in_text

        print("1:")
        # 変換前のテキストを分解して終了位置ごとにリストでまとめる
        nodes_set = [[] for _ in range(in_len)]
        for i in range(in_len):
            for prefix in self.trie.prefixes(in_text[i:]):
                nodes_set[i + len(prefix) - 1].append(prefix)
        print(nodes_set)

        # ノード作成
        connect_node = []
        for i in range(len(nodes_set)):
            min_cost = COST_TMP_MAX
            min_words = []

            for word in nodes_set[i]:
                node_len = len(word)
                connect_index = i - node_len
                for word_id in self.id_list(word):
                    words_tmp = [word_id]
                    print_str = self.words[word_id]["w"]

                    cost_tmp = self.score(words_tmp)
                    if cost_tmp < min_cost:
                        min_cost = cost_tmp
                        min_words = words_tmp
                        print("[%d] : cost %f / %s" % (i, min_cost, print_str))

                    if connect_index > 0:
                        words_tmp = connect_node[connect_index]["words"] + [word_id]
                        print_str = ""
                        for wordid in words_tmp:
                            print_str += self.words[wordid]["w"]

                        cost_tmp = self.score(words_tmp)
                        if cost_tmp < min_cost:
                            min_cost = cost_tmp
                            min_words = words_tmp
                            print("[%d] : cost %f / %s" % (i, min_cost, print_str))

            connect_node.append({"cost": min_cost, "words": min_words})

    def score(self, id_list):
        """ スコア取得

        :param id_list:
        :return:
        """
        emb = np.zeros(1, dtype=np.float32)
        emb_tmp = torch.zeros(768, device="cuda")
        emb_zeros = torch.zeros(768, device="cuda")
        for word_id in id_list:
            x_in = torch.from_numpy(self.words[word_id]["e"]).to("cuda")
            x_emb = self.b_model(x_in)
            emb_tmp = torch.add(x_emb, emb_tmp)
        x_emb = torch.cat([emb_tmp, emb_zeros])
        y = self.e_model(x_emb)
        score = self.criterion(y, x_emb)
        return score

    def id_list(self, word):
        """ TrieIDリスト取得

        :param word:
        :return:
        """
        ret = []
        for word_id in self.trie.get(word):
            ret.append(word_id[0])
        return ret

    def vector(self, word_id):
        """ DNN用単語ベクトル取得

        :param word_id:
        :return:
        """
        id_list = self.words[word_id]
        return self.word_info(id_list[1], id_list[2], id_list[3])


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model_path', help='model path', required=True)
    arg_parser.add_argument('-t', '--text', help='pre text', required=True)
    args = arg_parser.parse_args()

    converter = Converter(args.model_path)
    ret = converter(args.text)
    print(ret)


if __name__ == "__main__":
    main()


