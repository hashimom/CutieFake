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
import argparse
import os
import json
import csv
import glob


class PreBuilder:
    def __init__(self, in_dir, out_dir):
        """ モデルビルド用CSVファイル作成（pre処理）

        :param in_dir:
        :param out_dir:
        """
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.in_dir = in_dir

    def __call__(self):
        """ モデルビルド用CSVファイル作成

        :return:
        """
        with open(self.out_dir + "/word_link.csv", 'w', encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator='\n')
            file_list = glob.glob(self.in_dir + "/*.json")
            for file in file_list:
                with open(file, encoding="utf-8") as in_json:
                    info = json.load(in_json)

                    for chunk in info["Chunks"]:
                        word_is = False

                        if chunk["Independent"] is not None:
                            words = chunk["Independent"] + chunk["Ancillary"]
                            word_str = ""
                            for word in words:
                                if word["position"][0] != "特殊":
                                    word_str = word_str + word["surface"] + " "
                                    word_is = True
                            word_str = word_str.strip()

                            if word_is:
                                if chunk["Link"] is not None:
                                    link_str = ""
                                    for link in chunk["Link"]:
                                        link_str = link_str + link["surface"] + " "
                                    link_str = link_str.strip()
                                    writer.writerow([word_str, link_str])
                                else:
                                    writer.writerow([word_str, None])


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', nargs='?', help='input json file path', required=True)
    arg_parser.add_argument('-o', nargs='?', help='output directory name', required=True)
    args = arg_parser.parse_args()

    builder = PreBuilder(args.i, args.o)
    builder()


if __name__ == "__main__":
    main()
