#!/usr/bin/env python3

"""
Simple autoencoder for MNIST digits.
"""

import tensorflow as tf
import numpy as np
import string

def random_run_string():
    """ ugly to separate results from different runs with same params """
    s = string.ascii_uppercase
    r = ''
    for i in np.random.random_integers(0, 25, 6).tolist():
        r = r + s[i]
    return r

def mnist_labelled(flatten=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate( [x_train, x_test], axis=0 )
    Y = np.concatenate( [y_train, y_test], axis=0 )
    assert X.shape[1] == 28 and X.shape[2] == 28, "MNIST sample must be 28x28"
    X = np.reshape(X, [-1, 784]) # normalizing helps
    X = X / 255.0
    return X, Y

if __name__ == "__main__":
    data, _ = mnist_labelled()
    hidden_units = 128
    prefix = 'MNIST_Autoencoder_'
    def get_param_string():
        return prefix + "hidden_units_" + str(hidden_units) + "_run_" + random_run_string()

    LOG_NAME = get_param_string()

    with tf.Session() as sess:
        """ Model: Linear transform to smaller space, and then decode. """
        """ encode """
        X = tf.placeholder( tf.float32, shape=[None, 784] )
        W = tf.Variable( tf.random_uniform(shape=[784, hidden_units], dtype=tf.float32) )
        b = tf.Variable( tf.random_uniform(dtype=tf.float32, shape=[1]) )
        encoding = tf.matmul(X, W) + b
        """ decode """
        W2 = tf.Variable( tf.random_uniform(shape=[hidden_units, 784], dtype=tf.float32) )
        b2 = tf.Variable( tf.random_uniform(dtype=tf.float32, shape=[1]) )
        X2 = tf.matmul(encoding, W2) + b2

        """ summaries """
        loss = tf.losses.mean_squared_error( X, X2 )
        loss_op = tf.summary.scalar('loss', loss)
        write_loss = tf.summary.merge( [loss_op] )
        input_img = tf.placeholder(tf.float32, shape=[None,28,28,1])
        input_org = tf.placeholder(tf.float32, shape=[None,28,28,1])
        img_op = tf.summary.image('estimated', input_img, max_outputs=10)
        org_op = tf.summary.image('original', input_org, max_outputs=10)
        write_img = tf.summary.merge([img_op, org_op])
        training_write_loss = tf.summary.FileWriter('train/' + LOG_NAME)
        """ training """
        train_op = tf.train.AdamOptimizer(1.0).minimize(loss)
        
        tf.global_variables_initializer().run()
        
        for epoch in range(0, 220):
            """ Train on samples 0->60,000 """
            for i in range(0, 60):
                batch = data[1000*i:1000*i+1000]
                loss_val, summary_val = sess.run([train_op, write_loss], feed_dict={X: batch})
                training_write_loss.add_summary(summary_val, epoch*60+i)
            """ Test with samples 65,000 -> 66,000 """
            IMAGE = sess.run([X2], feed_dict={X: np.reshape(data[65000:66000], (-1,784))})
            IMAGE = np.reshape(IMAGE, [-1,28,28,1])
            img_val = sess.run(write_img, feed_dict={input_org: np.reshape(data[65000:66000],[-1,28,28,1]), input_img: IMAGE })
            training_write_loss.add_summary(img_val, epoch)

