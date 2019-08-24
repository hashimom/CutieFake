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
import csv
import argparse
import os
import numpy as np
import tensorflow as tf
from egoisticlily.modelmaker.wordvector import WordVector
from egoisticlily.modelmaker.wordholder import WordHolder

WORD_PHRASE_NUM = 4
WORD_ID_BIT_NUM = 16


class Builder:
    def __init__(self, word_file, link_file, output_dir):
        # DNN define
        hid_dim = 64
        z_dim = 16
        hid_num = 1

        self.word_holder = WordHolder(word_file)
        type1_cnt, type2_cnt = self.word_holder.type_list_cnt()
        self.link_list = []
        with open(link_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                self.link_list.append(row)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.out_dir = output_dir

        self.model = WordVector(hid_dim, z_dim, hid_num, WORD_ID_BIT_NUM, type1_cnt, type2_cnt, WORD_PHRASE_NUM)
        self.optimizer = tf.optimizers.Adam()

    def __call__(self, epoch_num, batch_size):
        for i in range(epoch_num):
            batch_list = self.make_batch_list(batch_size)
            batch = self.make_batch(batch_list, batch_size)
            with tf.GradientTape() as tape:
                score, y = self.model.score(batch)
            grads = tape.gradient(score, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            print(str(i + 1) + ":  score= " + str(score))

            self.update_word_id(batch_list, y)
        self.word_holder.save(self.out_dir)

        tf.saved_model.save(self.model, self.out_dir + "/dnn/")

    def make_batch_list(self, batch_size):
        batch_list = []
        rand_list = np.random.randint(0, len(self.link_list), batch_size)
        for i in rand_list:
            batch_list.append(self.link_list[i])
        return batch_list

    def make_batch(self, batch_list, batch_size):
        batch = np.empty((batch_size, 320), dtype="float")
        for i, x in enumerate(batch_list):
            word_id = np.empty((WORD_PHRASE_NUM, 80), dtype="float")
            for j in range(WORD_PHRASE_NUM):
                if not x[j] in self.word_holder.word_list:
                    self.word_holder.regist(x[j], x[j], "未定義語", "その他")
                word_id[j] = self.word_holder(x[j])
            batch[i] = word_id.reshape(1, 320)
        return batch

    def update_word_id(self, batch_list, y):
        for i, link in enumerate(batch_list):
            y_word_ary = np.reshape(y[i], (WORD_PHRASE_NUM, 80))
            for j, word in enumerate(link):
                new_id = 0
                y_word = y_word_ary[j][:WORD_ID_BIT_NUM]
                y_mean = tf.reduce_mean(y_word)
                for val in y_word:
                    if val > y_mean:
                        new_id += 1
                    new_id = new_id << 1
                self.word_holder.word_list[word]["vec_id"] = int(new_id)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-l', nargs='?', help='input link file', required=True)
    arg_parser.add_argument('-w', nargs='?', help='input word file', required=True)
    arg_parser.add_argument('-o', nargs='?', help='output model directory', required=True)
    args = arg_parser.parse_args()

    builder = Builder(args.w, args.l, args.o)
    builder(200, 500)


if __name__ == "__main__":
    main()



