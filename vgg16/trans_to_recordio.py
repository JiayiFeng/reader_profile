import os
import numpy as np
import sys
import time
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

batch_size = 12

train_reader = paddle.batch(paddle.dataset.flowers.train(), batch_size)

def trans_to_recordio(reader, dst_file_name):
    feeder = fluid.DataFeeder(
        feed_list=[fluid.layers.data(
            name='image', shape=[3, 224, 224], dtype='float32'),
            fluid.layers.data(
            name='label', shape=[1], dtype='int64')],
        place=fluid.CPUPlace())
    fluid.recordio_writer.convert_reader_to_recordio_file(
        dst_file_name, reader, feeder, compressor=core.RecordIOWriter.Compressor.NoCompress)


if __name__ == '__main__':
    print("Start transforming data to recordio...")
    trans_to_recordio(reader=train_reader,
                      dst_file_name="./train.recordio")
    print("Complete transforming.")
