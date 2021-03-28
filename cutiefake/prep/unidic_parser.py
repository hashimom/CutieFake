# -*- coding: utf-8 -*-
import numpy as np


class UnidicParser:
    _POS_LIST = [
        [
            "名詞",
            "代名詞",
            "形状詞",
            "連体詞",
            "副詞",
            "接続詞",
            "感動詞",
            "動詞",
            "形容詞",
            "助動詞",
            "助詞",
            "接頭辞",
            "接尾辞",
            "記号",
            "補助記号",
            "空白",
            "_BOS",
            "_EOS"
        ],
        [
            "名詞-普通名詞",
            "名詞-固有名詞",
            "名詞-数詞",
            "名詞-助動詞語幹",
            "形状詞-一般",
            "形状詞-タリ",
            "形状詞-助動詞語幹",
            "感動詞-一般",
            "感動詞-フィラー",
            "動詞-一般",
            "動詞-非自立可能",
            "形容詞-一般",
            "形容詞-非自立可能",
            "助詞-格助詞",
            "助詞-副助詞",
            "助詞-係助詞",
            "助詞-接続助詞",
            "助詞-終助詞",
            "助詞-準体助詞",
            "接尾辞-名詞的",
            "接尾辞-形状詞的",
            "接尾辞-動詞的",
            "接尾辞-形容詞的",
            "記号-一般",
            "記号-文字",
            "補助記号-一般",
            "補助記号-句点",
            "補助記号-読点",
            "補助記号-括弧開",
            "補助記号-括弧閉",
            "_STAR"
        ],
        [
            "名詞-普通名詞-一般",
            "名詞-普通名詞-サ変可能",
            "名詞-普通名詞-形状詞可能",
            "名詞-普通名詞-サ変形状詞可能",
            "名詞-普通名詞-副詞可能",
            "名詞-普通名詞-助数詞可能",
            "名詞-固有名詞-一般",
            "名詞-固有名詞-人名",
            "名詞-固有名詞-地名",
            "名詞-固有名詞-組織名",
            "接尾辞-名詞的-一般",
            "接尾辞-名詞的-サ変可能",
            "接尾辞-名詞的-形状詞可能",
            "接尾辞-名詞的-副詞可能",
            "接尾辞-名詞的-助数詞",
            "_STAR"
        ],
        [
            "名詞-固有名詞-人名-一般",
            "名詞-固有名詞-人名-姓",
            "名詞-固有名詞-人名-名",
            "名詞-固有名詞-地名-一般",
            "名詞-固有名詞-地名-国",
            "_STAR"
        ]
    ]

    def pos_id(self, pos):
        pos_id_1 = self._POS_LIST[0].index(pos[0])

        pos_id_2 = self._POS_LIST[1].index("_STAR")
        if pos[1] != "*":
            pos_id_2 = self._POS_LIST[1].index(pos[0] + "-" + pos[1])

        pos_id_3 = self._POS_LIST[2].index("_STAR")
        if pos[2] != "*":
            pos_id_3 = self._POS_LIST[2].index(pos[0] + "-" + pos[1] + "-" + pos[2])

        pos_id_4 = self._POS_LIST[3].index("_STAR")
        if pos[3] != "*":
            pos_id_4 = self._POS_LIST[3].index(pos[0] + "-" + pos[1] + "-" + pos[2] + "-" + pos[3])

        return [pos_id_1, pos_id_2, pos_id_3, pos_id_4]

    def bos(self):
        return ['_BOS', [self._POS_LIST[0].index("_BOS"), self._POS_LIST[1].index("_STAR"),
                self._POS_LIST[2].index("_STAR"), self._POS_LIST[3].index("_STAR")]]

    def eos(self):
        return [self._POS_LIST[0].index("_EOS"), self._POS_LIST[1].index("_STAR"),
                self._POS_LIST[2].index("_STAR"), self._POS_LIST[3].index("_STAR")]
