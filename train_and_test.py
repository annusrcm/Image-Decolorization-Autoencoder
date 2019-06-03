import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime

from logger import Logger
from config import Config

config = Config()


def autoencoder(inputs):

    # Encoder

    net = tf.layers.conv2d(inputs, 128, 2, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='same')
    Logger.log("Encoder Network shape is : {}".format(net.shape))

    # Decoder

    net = tf.image.resize_nearest_neighbor(net, tf.constant([129, 129]))
    net = tf.layers.conv2d(net, 1, 2, activation=None, name='outputOfAuto')

    Logger.log("Decoder Network shape is : {}".format(net.shape))
    return net


def get_training_data():
    dataset_source = []
    for file in os.listdir(config.color_path):
        img = cv2.imread(os.path.join(config.color_path, file))
        dataset_source.append(np.array(img))
    dataset_source = np.asarray(dataset_source)
    Logger.log("Shape of source data : {}".format(dataset_source.shape))

    dataset_tar = []
    for file in os.listdir(config.gray_path):
        img = cv2.imread(os.path.join(config.gray_path, file), 0)
        dataset_tar.append(np.array(img))
    dataset_target = np.asarray(dataset_tar)
    Logger.log("Shape of target data : {}".format(dataset_target.shape))
    # add extra axis to maintain data uniformity
    dataset_target = dataset_target[:, :, :, np.newaxis]
    Logger.log("Shape of target data after adding new axis: {}".format(dataset_target.shape))

    return dataset_source, dataset_target


def training():
    start = datetime.now()
    saving_path = config.model_folder
    saver_ = tf.train.Saver(max_to_keep=3)

    dataset_source, dataset_target = get_training_data()
    batch_img = dataset_source[0:config.batch_size]
    batch_out = dataset_target[0:config.batch_size]

    num_batches = num_images // config.batch_size
    Logger.log("Number of batches : {}".format(num_batches))

    for ep in range(config.epoch_num):
        batch_size = 0
        for batch_n in range(num_batches):  # batches loop

            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img, ae_target: batch_out})
            Logger.log("Epoch: {} ------------------- cost = {:.5f}".format((ep + 1), c))

            batch_img = dataset_source[batch_size: batch_size + config.batch_size]
            batch_out = dataset_target[batch_size: batch_size + config.batch_size]

            batch_size += config.batch_size

        saver_.save(sess, saving_path, global_step=ep)

    sess.close()
    Logger.log("TRAINING FINISHED in {} seconds".format(datetime.now()-start))


# TESTING
def testing():
    start = datetime.now()
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    model = config.model_folder + "-" + str(config.epoch_num - 1)
    saver.restore(sess, model)

    test_data = []
    for file in os.listdir(config.test_gray_path):
        try:
            print("Reading {} for prediction".format(file))
            img = cv2.imread(os.path.join(config.test_gray_path, file))
            test_data.append(np.array(img))
        except Exception as e:
            Logger.log(e)
            Logger.log("Could not read : {}".format(os.path.join(config.test_gray_path, file)))

    test_dataset = np.asarray(test_data)

    # Running the test data on the autoencoder
    batch_imgs = test_dataset
    gray_imgs = sess.run(ae_outputs, feed_dict={ae_inputs: batch_imgs})

    for i in range(gray_imgs.shape[0]):
        output_fn = 'gen_gray_' + str(i) + '.jpeg'
        cv2.imwrite(os.path.join(config.test_output, output_fn), gray_imgs[i])

    Logger.log("TESTING FINISHED in {} seconds".format(datetime.now()-start))


if __name__ == "__main__":
    num_images = len(os.listdir(config.color_path))
    ae_inputs = tf.placeholder(tf.float32, (None, 128, 128, 3), name='inputToAuto')
    ae_target = tf.placeholder(tf.float32, (None, 128, 128, 1))

    ae_outputs = autoencoder(ae_inputs)
    lr = 0.001  # learning rate

    loss = tf.reduce_mean(tf.square(ae_outputs - ae_target))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # Intialize the network
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    training()
    testing()
