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
                self.words.append(row[0])
                trie_keys.append(row[1])
                trie_values.append([i, int(row[2]), int(row[3]), int(row[4])])

        self.trie = marisa_trie.RecordTrie("<IIHH", zip(trie_keys, trie_values))
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
        print(self.trie.get("わ"))

        a = self.score(64539, 10, 24, 15036, 9, 9)
        print(a)

        # graph build
        # for i in range(len(in_text)):

    def score(self, w1_vec_id, w1_type1, w1_type2, w2_vec_id, w2_type1, w2_type2):
        w1_vec = self.word_info(w1_vec_id, w1_type1, w1_type2)
        w2_vec = self.word_info(w2_vec_id, w2_type1, w2_type2)
        return self.model.score(w1_vec + w2_vec)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', nargs='?', help='input json file path', required=True)
    arg_parser.add_argument('-m', nargs='?', help='output directory name', required=True)
    args = arg_parser.parse_args()

    converter = Converter(args.m)
    converter("わたしのなまえはなかのです")


if __name__ == "__main__":
    main()


