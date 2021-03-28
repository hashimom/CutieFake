# -*- coding: utf-8 -*-
import argparse
import pickle
import jaconv
import zipfile
import csv
from io import TextIOWrapper
from tqdm import tqdm
from multiprocessing import Process, Queue
from sudachipy import tokenizer
from sudachipy import dictionary


"""
class Parser(Process):
    def __init__(self, in_q, out_q, dict_ids):
        super().__init__()
        self._in_q = in_q
        self._out_q = out_q
        self._dict_ids = dict_ids
        self._unidic = UniDicUtil()

    def run(self):
        while not self._in_q.empty():
            text = self._in_q.get()
            self.parse(text)

    def parse(self, text):
        bos_id = self._unidic.bos()
        ret_data = [[bos_id[0], bos_id[1], bos_id[2], bos_id[3], 0]]

        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer.Tokenizer.SplitMode.C
        for m in tokenizer_obj.tokenize(text, mode):
            surface = m.surface()
            reading_form = jaconv.kata2hira(m.reading_form())
            pos_ary = m.part_of_speech()
            p_o_s = self._unidic.pos_id(pos_ary)
            id_list_str = str(p_o_s[0]) + "_" + str(p_o_s[1]) + "_" + str(p_o_s[2]) + "_" + str(p_o_s[3])
            key = surface + "__" + reading_form + "__" + id_list_str
            if key in self._dict_ids.keys():
                ret_data.append([p_o_s[0], p_o_s[1], p_o_s[2], p_o_s[3], self._dict_ids[key]])
            else:
                ret_data.append([p_o_s[0], p_o_s[1], p_o_s[2], p_o_s[3], 0])

        eos_id = self._unidic.eos()
        ret_data.append([eos_id[0], eos_id[1], eos_id[2], eos_id[3], 0])

        self._out_q.put(ret_data)
"""

class CreateTrainData:
    def __init__(self, in_path, dict_path, out_path):
        self._in_path = in_path
        self._dict_path = dict_path
        self._out_path = out_path

    def __call__(self, proc_num=2):
        with open(self._in_path, 'r') as f:
            # メモリ足りなくなる気がするけどひとまず暫定で全読みする処理・・・
            text_file = f.read().replace('\n', '')

        # 「。」毎にパースを行う ※それ以外は未対応
        in_q = Queue()
        in_cnt = 0
        for text in text_file.replace("。", "。____").split("____"):
            if len(text) != 0:
                in_q.put(text)
                in_cnt += 1

        dict_ids = self.create_dict_table()
        proc = []
        out_q = Queue()
        for i in range(proc_num):
            p = Parser(in_q, out_q, dict_ids)
            proc.append(p)
            p.start()

        write_data = []
        for _ in tqdm(range(in_cnt)):
            write_data.append(out_q.get())

        with open(self._out_path, 'wb') as f:
            pickle.dump(write_data, f)

        for p in proc:
            p.join()

    def create_dict_table(self):
        ret_dict = {}
        unidic_util = UniDicUtil()

        with zipfile.ZipFile(self._dict_path) as zp:
            with zp.open('egolili_dic.csv', 'r') as infile:
                reader = csv.reader(TextIOWrapper(infile, 'utf-8'))
                for row in reader:
                    try:
                        p_o_s = unidic_util.pos_id([row[3], row[4], row[5], row[6]])
                        id_list_str = str(p_o_s[0]) + "_" + str(p_o_s[1]) + "_" + str(p_o_s[2]) + "_" + str(p_o_s[3])
                        key = row[0] + "__" + row[1] + "__" + id_list_str
                        ret_dict[key] = row[7]
                    except ValueError:
                        pass

        return ret_dict


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_path', help='input text file path', required=True)
    arg_parser.add_argument('-d', '--dict_path', help='input dictionary path', default="in/egolilidic.zip")
    arg_parser.add_argument('-o', '--out_path', help='output pickle file path', default="in/training.pkl")
    arg_parser.add_argument('-p', '--proc_num', help='process num', default=2, type=int)
    args = arg_parser.parse_args()

    creator = CreateTrainData(args.in_path, args.dict_path, args.out_path)
    creator(args.proc_num)


if __name__ == "__main__":
    main()
