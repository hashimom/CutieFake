# -*- coding: utf-8 -*-
"""
 Copyright (c) 2020 Masahiko Hashimoto <hashimom@geeko.jp>

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
import time
import grpc
import cutiefake.proto.elily_pb2_grpc
import cutiefake.proto.elily_pb2
from concurrent import futures
from cutiefake.converter import Converter

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ELilyGateway(cutiefake.proto.elily_pb2_grpc.ELilyServiceServicer):
    def __init__(self, model):
        """ EgoisticLily gRPCサーバークラス

        :param model: モデルパス
        """
        self.converter = Converter(model)

    def Convert(self, request, context):
        """ 変換

        :param request:
        :param context:
        :return:
        """
        ret_str = self.converter(request.in_str)
        response = cutiefake.proto.elily_pb2.ConvertResp(status=200, out_str=ret_str)
        return response


def main():
    """ EgoisticLily gRPCサーバーモジュール

    :return:
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model path', required=True)
    arg_parser.add_argument('-p', '--port', help='server port number', default='50055')
    args = arg_parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cutiefake.proto.elily_pb2_grpc.add_ELilyServiceServicer_to_server(
        ELilyGateway(args.model), server)

    # portの設定
    port_str = '[::]:' + args.port
    server.add_insecure_port(port_str)
    server.start()
    print("EgoisticLily Server Start!  Port %d" % int(args.port))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)

    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()



