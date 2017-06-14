""" Convert labeled PNGs to TFRecords """

import argparse
import os

import tensorflow as tf
from tqdm import *
import cv2

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert():
    print('Writing TF records to ', FLAGS.output)
    writer = tf.python_io.TFRecordWriter(FLAGS.output)
    for labeled_dir in os.listdir(FLAGS.directory):
        print('Processing {} subdirectory'.format(labeled_dir))
        for image_name in tqdm(os.listdir(os.path.join(FLAGS.directory, labeled_dir))):
            image = cv2.imread(os.path.join(FLAGS.directory, labeled_dir, image_name))
            image_raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(int(labeled_dir)),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, help='Root dir with per-label subdirectories in it')
    parser.add_argument('--output', type=str, required=True, help='TFRecords output location')
    FLAGS, unparsed = parser.parse_known_args()
    convert()
