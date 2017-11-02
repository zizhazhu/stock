import tensorflow as tf
import numpy as np


def batch_generator(features, labels, batch_size=64):
    start_index = 0
    while start_index != features.shape[0]:
        end_index = start_index + batch_size if start_index + batch_size < features.shape[0] else features.shape[0]
        yield features[start_index:end_index], labels[start_index:end_index]
        start_index = end_index


class NeuralNetwork:

    def __init__(self, dimensions, epochs, learning_rate=1e-4, keep_prob=0.5, dtype=tf.float32):
        self.dtype = dtype
        self.epochs = epochs

        self.X = tf.placeholder(self.dtype, shape=(None, dimensions[0]), name='X')
        self.y = tf.placeholder(self.dtype, shape=(None, ), name='y')
        self.y_reshaped = tf.reshape(self.y, [-1, 1], name='label')

        layer1 = tf.layers.dense(self.X, dimensions[1], activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, dimensions[2], activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, dimensions[3], activation=tf.nn.relu)

        self.logits = tf.layers.dense(layer3, 1, activation=None)
        self.y_prob = tf.nn.sigmoid(self.logits)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_reshaped, logits=self.logits))

        self.data = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
        self.output = tf.cast(self.y_prob > 0.5, tf.int32)
        correct_prediction = tf.equal(self.y, tf.cast(self.output, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.Session()

    def fit(self, X, y):
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.epochs):
            cost = 0
            accuracy = None
            for batch_X, batch_y in batch_generator(X, y):
                cost, accuracy = self.sess.run([self.cross_entropy, self.accuracy], feed_dict={self.X: batch_X, self.y: batch_y})
            print("epoch:{} cost:{} train_accuracy:{}".format(i+1, cost, accuracy))

    def predict(self, X):
        output = self.sess.run(self.output, feed_dict={self.X: X})
        return output

    def predict_proba(self, X):
        output = np.zeros([X.shape[0], 2])
        output[:, 1] = self.sess.run(self.y_prob, feed_dict={self.X: X})
        output[:, 0] = 1 - output[:, 1]
        return output

    def score(self, X, y):
        y_pred = self.predict(X)
        right = 0
        for i in range(len(y)):
            if y_pred[i] == y[i]:
                right += 1
        return right / len(y)

