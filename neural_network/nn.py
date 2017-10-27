import tensorflow as tf


def batch_generator(features, labels, batch_size=64):
    start_index = 0
    while start_index != features.shape[0]:
        end_index = start_index + batch_size if start_index + batch_size < features.shape[0] else features.shape[0]
        yield features[start_index:end_index], labels[start_index:end_index]
        start_index = end_index


class NeuralNetwork:

    def __init__(self, dimensions, epochs, dtype=tf.float32):
        self.dtype = dtype
        self.epochs = epochs
        self.cost_tracer = []

        self.X = tf.placeholder(self.dtype, shape=(None, dimensions[0]))
        self.y = tf.placeholder(self.dtype, shape=(None, ))
        self.y_reshaped = tf.reshape(self.y, [-1, 1], name='label')

        layer1 = tf.layers.dense(self.X, dimensions[1], activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, dimensions[2], activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, dimensions[3], activation=tf.nn.relu)

        self.logits = tf.layers.dense(layer3, 1, activation=None)
        self.y_pred = tf.nn.sigmoid(self.logits)
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_reshaped, logits=self.logits))

        self.data = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.output = tf.cast(self.y_pred > 0.5, tf.int32)
        correct_prediction = tf.equal(self.y, tf.cast(self.output, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.Session()

    def fit(self, X, y):
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.epochs):
            accuracy = None
            for batch_X, batch_y in batch_generator(X, y):
                accuracy = self.sess.run([self.accuracy], feed_dict={self.X: batch_X, self.y: batch_y})
            print(accuracy)

    def predict(self, X):
        output = self.sess.run(self.output, feed_dict={self.X: X})
        return output

    def predict_proba(self, X):
        output = self.sess.run(self.y_pred, feed_dict={self.X: X})
        return output

    def score(self, X, y):
        y_pred = self.predict(X)
        right = 0
        for i in range(len(y)):
            if y_pred[i] == y[i]:
                right += 1
        return right / len(y)

