import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from ops import *
from utils import *
from scipy.misc.pilutil import toimage


mnist = input_data.read_data_sets('data')

height = 28
width = 28
channel = 1
batch_size = 100
hidden_dims = 64
recurrent_length = 2
out_hidden_dims = 64
out_recurrent_length = 2
learning_rate = 1e-3

train_step = int(mnist.train.num_examples / batch_size)
test_step = int(mnist.test.num_examples / batch_size)

print(mnist.train.num_examples, mnist.test.num_examples)

X = tf.placeholder(tf.float32, [None, height, width, channel])

hidden_layer = [conv2d(X, hidden_dims, [7, 7], "A", scope="conv_inputs")]
for i in range(recurrent_length):
    scope = 'CONV%d' % i
    hidden_layer.append(conv2d(hidden_layer[-1], hidden_dims, [3, 3], "B", scope=scope))

for i in range(out_recurrent_length):
    scope = 'CONV_OUT%d' % i
    hidden_layer.append(tf.nn.relu(conv2d(hidden_layer[-1], out_hidden_dims, [1, 1], "B", scope=scope)))

conv2d_out_logits = conv2d(hidden_layer[-1], 1, [1, 1], "B", scope='conv2d_out_logits')
output = tf.nn.sigmoid(conv2d_out_logits)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(conv2d_out_logits, X, name='loss'))
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)

new_grads_and_vars = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
optim = optimizer.apply_gradients(new_grads_and_vars)

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for epoch in range(300):
        # 1. train
        total_train_costs = []
        for i in range(train_step):
            batch = mnist.train.next_batch(batch_size)
            images = binarize(batch[0]).reshape([batch_size, height, width, channel])
            _, cost = sess.run([optim, loss], feed_dict={X: images})
            total_train_costs.append(cost)

        # 2. test
        total_test_costs = []
        for i in range(test_step):
            batch = mnist.test.next_batch(batch_size)
            images = binarize(batch[0]).reshape([batch_size, height, width, channel])
            cost = sess.run(loss, feed_dict={X: images})
            total_test_costs.append(cost)

        print(np.mean(total_train_costs), np.mean(total_test_costs))

        # stat.on_step(avg_train_cost, avg_test_cost)
        #
        # 3. generate samples
        samples = np.zeros((100, height, width, channel), dtype='float32')
        for i in range(height):
            for j in range(width):
                for k in range(channel):
                    next_sample = binarize(sess.run(output, {X: samples}))
                    samples[:, i, j, k] = next_sample[:, i, j, k]

        samples = samples.reshape((10, 10, height, width))
        samples = samples.transpose(1, 2, 0, 3)
        samples = samples.reshape((height * 10, width * 10))

        toimage(samples, cmin=0.1, cmax=1.0).save('sample/6epoch_%d.jpg' % (epoch+1))
