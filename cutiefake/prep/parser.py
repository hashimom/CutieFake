# -*- coding: utf-8 -*-
import argparse
from cutiefake.prep.unidic_parser import UnidicParser
from pyknp import KNP
from sudachipy import tokenizer
from sudachipy import dictionary


class Parser:
    def __init__(self):
        self._knp = KNP()
        self._tokenizer = dictionary.Dictionary().create()
        self._token_mode = tokenizer.Tokenizer.SplitMode.C
        self._uni_dic_parser = UnidicParser()

    def __call__(self, text):
        link_info = []
        phases = []

        result = self._parse(text)
        if result:
            for bst in result.bnst_list():
                chunk_len = 0
                for mrp in bst.mrph_list():
                    chunk_len += len(mrp.midasi)
                phases.append({"len": chunk_len, "link": bst.parent_id})

            chunk_words = []
            chunk_len = 0
            chunk_idx = 0
            for m in self._tokenizer.tokenize(text, self._token_mode):
                surface = m.surface()
                pos_ary = m.part_of_speech()
                pos_id = self._uni_dic_parser.pos_id(pos_ary)
                chunk_words.append([surface, pos_id])
                chunk_len += len(surface)
                if chunk_len >= phases[chunk_idx]["len"]:
                    phases[chunk_idx]["words"] = chunk_words
                    chunk_words = []
                    chunk_len = 0
                    chunk_idx += 1

            for index, chunk in enumerate(phases):
                is_exist = False
                for i, link_phase in enumerate(phases):
                    if index == link_phase["link"]:
                        link_info.append([phases[i]["words"], chunk["words"]])
                        is_exist = True

                if not is_exist:
                    link_info.append([self._uni_dic_parser.bos(), chunk["words"]])

        return link_info

    def _parse(self, text):
        result = None
        try:
            result = self._knp.parse(text)
        except Exception as e:
            print("text \"" + text + "\" " + str(e))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cutiefake_parser')
    parser.add_argument('-t', '--text', required=True)
    args = parser.parse_args()

    parser = Parser()
    ret = parser(args.text)
    if ret is not None:
        print(ret)
