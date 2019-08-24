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
import tensorflow as tf


class WordVector(tf.Module):
    def __init__(self, hid_dim, z_dim, hid_num, word_id_bit_num, type1_kind_num, type2_kind_num, phase_num):
        init_w = tf.random_normal_initializer()
        init_h = tf.zeros_initializer()
        self.hid_num = hid_num
        in_dim = (word_id_bit_num + type1_kind_num + type2_kind_num) * phase_num

        # encoder
        self.en_hid_w = []
        self.en_hid_h = []

        self.in_w = tf.Variable(init_w(shape=(in_dim, hid_dim), dtype="float64"))
        self.in_h = tf.Variable(init_h(shape=hid_dim, dtype="float64"))

        for i in range(hid_num):
            self.en_hid_w.append(tf.Variable(init_w(shape=(hid_dim, hid_dim), dtype="float64")))
            self.en_hid_h.append(tf.Variable(init_h(shape=hid_dim, dtype="float64")))

        self.z_out_w = tf.Variable(init_w(shape=(hid_dim, z_dim), dtype="float64"))
        self.z_out_h = tf.Variable(init_h(shape=z_dim, dtype="float64"))

        # decoder
        self.de_hid_w = []
        self.de_hid_h = []

        self.z_in_w = tf.Variable(init_w(shape=(z_dim, hid_dim), dtype="float64"))
        self.z_in_h = tf.Variable(init_h(shape=hid_dim, dtype="float64"))

        for i in range(hid_num):
            self.de_hid_w.append(tf.Variable(init_w(shape=(hid_dim, hid_dim), dtype="float64")))
            self.de_hid_h.append(tf.Variable(init_h(shape=hid_dim, dtype="float64")))

        self.out_w = tf.Variable(init_w(shape=(hid_dim, in_dim), dtype="float64"))
        self.out_h = tf.Variable(init_h(shape=in_dim, dtype="float64"))

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 320], dtype=tf.float64),))
    def score(self, x_in):
        # encoder
        h = tf.nn.relu(tf.matmul(x_in, self.in_w) + self.in_h)
        for i in range(self.hid_num):
            h = tf.nn.relu(tf.matmul(h, self.en_hid_w[i]) + self.en_hid_h[i])
        z = tf.nn.relu(tf.matmul(h, self.z_out_w) + self.z_out_h)

        # decoder
        h = tf.nn.relu(tf.matmul(z, self.z_in_w) + self.z_in_h)
        for i in range(self.hid_num):
            h = tf.nn.relu(tf.matmul(h, self.de_hid_w[i]) + self.de_hid_h[i])
        y = tf.matmul(h, self.out_w) + self.out_h

        score = tf.reduce_mean(tf.square(y - x_in))
        return score, y

