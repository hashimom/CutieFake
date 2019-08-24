# -*- coding: utf-8 -*-
"""
 Copyright (c) 2019 Masahiko Hashimoto <hashimom@geeko.jp>

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
import tensorflow as tf
from egoisticlily.words import Words


class Converter:
    def __init__(self, model_dir):
        self.words = []
        trie_keys = []
        trie_values = []
        model_path = os.path.abspath(model_dir)
        with open(model_path + "/words.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                self.words.append([row[0], int(row[2]), int(row[3]), int(row[4])])
                trie_keys.append(row[1])
                trie_values.append([i])

        self.trie = marisa_trie.RecordTrie("<I", zip(trie_keys, trie_values))
        self.model = tf.saved_model.load(model_dir + "/dnn/")

        self.word_info = Words()

    def __call__(self, in_text):
        # nodes build
        nodes = [[] for i in range(len(in_text))]
        for i in range(len(in_text)):
            for prefix in self.trie.prefixes(in_text[i:]):
                nodes[i+len(prefix)-1].append(prefix)
        nodes.insert(len(in_text), ["@E@"])
        nodes.insert(0, ["@S@"])
        print(nodes)

        min_cost = 0
        fix_str = ""
        node_min = 0
        pre_word_id = self.id_list("@S@")[0]
        for i in range(len(nodes)):
            # bi-gramは2文字以上から
            if i == 0:
                continue

            for node_min in nodes[i]:
                print("node_min: " + node_min)

                for same_word in self.id_list(node_min):
                    same_word_min_cost = 0
                    # print(str(word_id) + ": " + self.words[word_id][0])
                    score = self.score(pre_word_id, same_word)
                    if score < same_word_min_cost or same_word_min_cost == 0:
                        same_word_min_id = same_word
                        same_word_min_cost = score
                        same_word_min_str = self.words[same_word][0]

                    print(same_word_min_str)
                pre_word_id = same_word_min_id

        # a = self.score("私", "の")
        # print(a)

    def score(self, word1_id, word2_id):
        # 係り受け解析部 ※未実装
        non_id_vec = self.vector(self.id_list("@N@")[0])
        vec = np.vstack((non_id_vec, non_id_vec))

        vec = np.vstack((vec, self.vector(word1_id)))
        vec = np.vstack((vec, self.vector(word2_id)))
        vec = vec.reshape(1, 320)
        return self.model.score(vec)[0]

    def id_list(self, word):
        ret = []
        for word_id in self.trie.get(word):
            ret.append(word_id[0])
        return ret

    def vector(self, word_id):
        id_list = self.words[word_id]
        return self.word_info(id_list[1], id_list[2], id_list[3])


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', nargs='?', help='output directory name', required=True)
    args = arg_parser.parse_args()

    converter = Converter(args.m)
    converter("わたしのなまえはなかのです")


if __name__ == "__main__":
    main()


