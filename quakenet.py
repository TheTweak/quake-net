import tensorflow as tf

# 15 classes, representing the main locations on DM6 map:
# 0) arena
# 1) green armor pillars
# 2) lightning gun
# 3) green armor ssg
# 4) ssg
# 5) super machine gun
# 6) arena balcony center
# 7) nail gun
# 8) heavy armor
# 9) bridge (above heavy armor)
# 10) rocket
# 11) quad
# 12) mega
# 13) green armor rail
# 14) rail
NUM_CLASSES = 15

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 72
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

