import argparse

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
NUM_CHANNELS = 3
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(x, keep_prob):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 72, 128, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([18 * 32 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 18 * 32 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv


def x_entropy_op(labels, predictions):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions))


def train_op(loss, learning_rate=1e-4):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    image.set_shape([IMAGE_PIXELS])
    label = tf.one_hot(features['label'], NUM_CLASSES, 1.0, 0.0)
    return image, label


def inputs(filename, batch_size, num_epochs):
    with tf.name_scope('input'):
        filname_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image, label = read_and_decode(filname_queue)
        images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=1,
                                                       capacity=1000 + 3 * batch_size, min_after_dequeue=1000,
                                                       allow_smaller_final_batch=True)
        return images, sparse_labels


def accuracy_op(y_conv, labels):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def run():
    with tf.Graph().as_default():
        images, labels = inputs(FLAGS.input_filename, FLAGS.batch_size, FLAGS.num_epochs)
        sess = tf.Session()

        coord = tf.train.Coordinator()

        keep_prob = tf.placeholder(tf.float32)
        y_conv = inference(images, keep_prob)
        x_entropy = x_entropy_op(labels, y_conv)
        accuracy = accuracy_op(y_conv, labels)
        train = train_op(x_entropy)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        is_training = FLAGS.mode == 'train'
        train_writer = tf.summary.FileWriter('E:/tensorboard_log/v1/{}/'.format(FLAGS.mode), sess.graph)
        saver = tf.train.Saver()
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('cross_entropy', x_entropy)
        merged = tf.summary.merge_all()
        if not is_training:
            saver.restore(sess, FLAGS.model_dir)
        try:
            step = 0
            while not coord.should_stop():
                if is_training:
                    summary, _ = sess.run([merged, train], feed_dict={keep_prob: 0.5})
                if not is_training or step % 100 == 0:
                    summary, _ = sess.run([merged, accuracy], feed_dict={keep_prob: 1.0})
                train_writer.add_summary(summary, step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop()

        coord.join(threads)
        if is_training:
            saver.save(sess, FLAGS.model_dir)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str, required=False, help='Input TFRecords file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where to save the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run: train or test')
    FLAGS, unparsed = parser.parse_known_args()
    run()
