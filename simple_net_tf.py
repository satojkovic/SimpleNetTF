#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


def next_batch(X_train, y_train, batch_size):
    mask = np.random.choice(X_train.shape[0], batch_size)
    batch_xs, batch_ys = X_train[mask], y_train[mask]
    return batch_xs, batch_ys


def one_hot_encoding(target):
    n_classes = np.max(target) + 1
    one_hot = [list((np.arange(n_classes) == t).astype(np.int))
               for t in target]
    return np.array(one_hot)


def main():
    # load an input data
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, one_hot_encoding(digits.target))
    print('Train data: {}'.format(X_train.shape))
    print('Train labels: {}'.format(y_train.shape))
    print('Test data: {}'.format(X_test.shape))
    print('Test labels: {}'.format(y_test.shape))

    # parameters
    learning_rate = 0.001
    batch_size = 70
    training_epochs = 100000
    display_step = 10

    # network parameters
    n_input = digits.data.shape[1]
    n_hidden = 70
    n_output = len(digits.target_names)

    # tf Graph input
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_output])

    # model definition
    def SimpleNet(x, weights, biases):
        # hidden layer with relu activation
        with tf.name_scope('hidden_layer') as scope:
            hidden_layer = tf.add(tf.matmul(x, weights['W1']), biases['b1'])
            hidden_layer = tf.nn.relu(hidden_layer)
        # output layer with softmax function
        with tf.name_scope('output_layer') as scope:
            out = tf.add(tf.matmul(hidden_layer, weights['W2']), biases['b2'])
            out = tf.nn.softmax(out)
        return out

    # store layers weights and bias
    weights = {
        'W1': tf.Variable(
            tf.random_normal(shape=[n_input, n_hidden]), name='weights'),
        'W2': tf.Variable(
            tf.random_normal(shape=[n_hidden, n_output]), name='weights')
    }
    biases = {'b1': tf.Variable(
        tf.random_normal(shape=[n_hidden]), name='biases'),
              'b2': tf.Variable(
                  tf.random_normal(shape=[n_output]), name='biases')}

    # construct model
    pred = SimpleNet(x, weights, biases)

    # define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=y))
    tf.summary.scalar('entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy)

    # initialize the variables
    init = tf.global_variables_initializer()

    # launch the graph
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            '/tmp/SimpleNetLog', graph=sess.graph)
        sess.run(init)
        summary_op = tf.summary.merge_all()

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = X_train.shape[0] // batch_size
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(X_train, y_train, batch_size)
                feed_dict = {x: batch_xs, y: batch_ys}
                _, c = sess.run([train_step, cross_entropy],
                                feed_dict=feed_dict)

                # model evaluation
                avg_cost += c / total_batch
            # Display logs
            if epoch % display_step == 0:
                print("Epoch %04d" % (epoch + 1),
                      "cost = {:.9f}".format(avg_cost))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch)
        print('Training finished.')

        # test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))


if __name__ == '__main__':
    main()
