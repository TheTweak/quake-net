""" Label images from video file manually """

import argparse
import os

import cv2
import quakenet
import random
import string


FLAGS = None


def random_string(length=32):
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(length)])


def save_image(pixels, label):
    root_dir, _ = os.path.split(FLAGS.video_location)
    os.makedirs(os.path.join(root_dir, label), exist_ok=True)
    cv2.imwrite(os.path.join(root_dir, label, random_string() + '.png'), pixels)


def run():
    keys = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'),
            ord('9'), ord('a'), ord('s'), ord('d'), ord('f'), ord('g')]
    cap = cv2.VideoCapture(FLAGS.video_location)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('frame', frame)

        while True:
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key in keys:
                print('ok')
                label = keys.index(pressed_key)
                resized_frame = cv2.resize(frame, (quakenet.IMAGE_WIDTH, quakenet.IMAGE_HEIGHT))
                save_image(resized_frame, str(label))
                break
            elif pressed_key == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-location', type=str, required=True, help='Input video file')
    FLAGS, unparsed = parser.parse_known_args()
    run()
