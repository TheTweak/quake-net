""" Split video into png files """

import argparse
import os

import cv2
import quakenet
import random
import string


FLAGS = None


def random_string(length=32):
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(length)])


def split_video():
    root_dir, video_file_name = os.path.split(FLAGS.video_location)
    label = video_file_name.split('.')[0]
    os.makedirs(os.path.join(root_dir, label))
    cap = cv2.VideoCapture(FLAGS.video_location)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (quakenet.IMAGE_WIDTH, quakenet.IMAGE_HEIGHT))
        cv2.imwrite(os.path.join(root_dir, label, random_string() + '.png'), resized_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-location', type=str, required=True, help='Input video file')
    FLAGS, unparsed = parser.parse_known_args()
    split_video()
