import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc.pilutil import toimage


def conv2d(input, num_output, kernel_shape, mask_type, scope="conv2d"):

    with tf.variable_scope(scope):

        kernel_h, kernel_w = kernel_shape
        assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width should be odd number"
        center_h = kernel_h // 2
        center_w = kernel_w // 2

        channel_len = input.get_shape()[-1]
        mask = np.ones((kernel_h, kernel_w, channel_len, num_output), dtype=np.float32)
        mask[center_h, center_w+1:, :, :] = 0.
        mask[center_h+1:, :, :, :] = 0.
        if mask_type == 'A':
            mask[center_h, center_w, :, :] = 0.

        weight = tf.get_variable("weight", [kernel_h, kernel_w, channel_len, num_output], tf.float32, tf.contrib.layers.xavier_initializer())
        weight *= tf.constant(mask, dtype=tf.float32)

        value = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME', name='value')
        bias = tf.get_variable("bias", [num_output], tf.float32, tf.zeros_initializer)
        output = tf.nn.bias_add(value, bias, name='output')

        print('[conv2d_%s] %s : %s %s -> %s %s' % (mask_type, scope, input.name, input.get_shape(), output.name, output.get_shape()))

        return output


def binarize(images):
    return (np.random.uniform(size=images.shape) < images).astype('float32')


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

hidden_layer = [conv2d(X, hidden_dims, [7, 7], "A", scope="conv_A")]
for i in range(recurrent_length):
    hidden_layer.append(conv2d(hidden_layer[-1], hidden_dims, [3, 3], "B", scope='conv_B_%d' % i))

for i in range(out_recurrent_length):
    hidden_layer.append(tf.nn.relu(conv2d(hidden_layer[-1], out_hidden_dims, [1, 1], "B", scope='relu_B_%d' % i)))

y_logits = conv2d(hidden_layer[-1], 1, [1, 1], "B", scope='y_logits')
y_ = tf.nn.sigmoid(y_logits)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_logits, X, name='loss'))
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)

new_grads_and_vars = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
optim = optimizer.apply_gradients(new_grads_and_vars)

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for epoch in range(300):
        # train
        total_train_costs = []
        for i in range(train_step):
            batch = mnist.train.next_batch(batch_size)
            images = binarize(batch[0]).reshape([batch_size, height, width, channel])
            _, cost = sess.run([optim, loss], feed_dict={X: images})
            total_train_costs.append(cost)

        # test
        total_test_costs = []
        for i in range(test_step):
            batch = mnist.test.next_batch(batch_size)
            images = binarize(batch[0]).reshape([batch_size, height, width, channel])
            cost = sess.run(loss, feed_dict={X: images})
            total_test_costs.append(cost)

        print(np.mean(total_train_costs), np.mean(total_test_costs))

        # generate samples
        samples = np.zeros((100, height, width, channel), dtype='float32')
        for i in range(height):
            for j in range(width):
                for k in range(channel):
                    next_sample = binarize(sess.run(y_, {X: samples}))
                    samples[:, i, j, k] = next_sample[:, i, j, k]

        samples = samples.reshape((10, 10, height, width))
        samples = samples.transpose(1, 2, 0, 3)
        samples = samples.reshape((height * 10, width * 10))

        toimage(samples, cmin=0.1, cmax=1.0).save('sample/epoch_%d.jpg' % (epoch+1))
