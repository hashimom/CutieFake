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


class Converter:
    def __init__(self, model_dir):
        self.words = []
        model_path = os.path.abspath(model_dir)
        with open(model_path + "/words.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                self.words.append(row[0])

        self.trie = marisa_trie.Trie()
        self.trie.load(model_path + "/words.marisa")

    def __call__(self, in_text):
        text = []
        for s in in_text:
            text.append(s)

        print(text)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', nargs='?', help='input json file path', required=True)
    arg_parser.add_argument('-m', nargs='?', help='output directory name', required=True)
    args = arg_parser.parse_args()

    converter = Converter(args.m)
    converter("たしかにでんげんいれた")


if __name__ == "__main__":
    main()


