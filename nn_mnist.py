import gzip
import cPickle

import tensorflow as tf
import numpy as np



# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
test_x, test_y = test_set
valid_x, valid_y = valid_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print train_y[57]



# TODO: the neural net!!
train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)

# np.random.shuffle(train_set)  # we shuffle the data
# x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
# y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

# W2 = tf.Variable(np.float32(np.random.rand(5, 5)) * 0.1)
# b2 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W3 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b3 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h2 = tf.nn.sigmoid(tf.matmul(h, W2) + b2)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.matmul(h, W3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
evaluacion = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(evaluacion, tf.float32))*100
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # learning rate: 0.5
square = tf.reduce_mean(tf.squared_difference(y_,y))
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
batch_x_valid = valid_x
batch_y_valid = valid_y
resultados= []

for epoch in xrange(10):
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    validation_error = sess.run(loss, feed_dict={x: batch_x_valid, y_:batch_y_valid})

    print "Epoch #:", epoch, "Error: ", validation_error
    square_error = sess.run(square, feed_dict={x: batch_x_valid , y_: batch_y_valid})
    accuracy_number = sess.run(accuracy, feed_dict={x: batch_x_valid, y_: batch_y_valid})
    print square_error, " acu --> ", accuracy_number
    resultados.append([square_error, accuracy_number])

    # result = sess.run(y, feed_dict={x: batch_xs})
    # for b, r in zip(batch_ys, result):
    #     print b, "-->", r
    # print "----------------------------------------------------------------------------------"


plt.plot(resultados)
plt.show()

print(accuracy.eval(feed_dict={x: test_x, y_: test_y}, session=sess))


