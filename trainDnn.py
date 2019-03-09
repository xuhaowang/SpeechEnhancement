'''
MMSE-based Fine-tuning
'''
import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import time

rng = np.random.RandomState(0)
random_state = 42

class Layer:

    def __init__(self, in_dim, out_dim ,layer_number, function=lambda x: x):
        if(layer_number < 3):
            self.W = tf.Variable(np.zeros([in_dim, out_dim]).astype('float32'), name='rbm_w')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'), name = 'rbm_h')
            self.v = tf.Variable(np.zeros([in_dim]).astype('float32'), name = 'rbm_v')
        else:
            self.W = tf.Variable(rng.uniform(low = -0.1, high = 0.1, size=(in_dim, out_dim)).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))

        self.function = function


    def create_saver(self, name = 'rbm'):
        saver = tf.train.Saver({name + '_w' : self.W,  name + '_v' : self.v,  name + '_h': self.b})
        return saver


    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z

#define computing graph
def prepare_data(file_path):
    all_data = np.zeros([1,257])
    #all_data = np.array(all_data)
    file_names = os.listdir(file_path)
    for name in file_names:
        full_path = file_path + name
        data = scio.loadmat(full_path)
        data = np.array(data['htkdata']).transpose()
        all_data = np.vstack((all_data, data))

    return all_data[1:all_data.shape[0]]

noisy_data_path = "DataSet/train/noisy_speech/"
clean_data_path = "DataSet/train/clean_speech/"

input_data = prepare_data(file_path = noisy_data_path)
label_data = prepare_data(file_path = clean_data_path)
#print(input_data.shape[1])

layers = [Layer(input_data.shape[1], 2048, 0, tf.nn.sigmoid), Layer(2048, 2048, 1,  tf.nn.sigmoid), Layer( 2048, 2048, 2, tf.nn.sigmoid), Layer(in_dim = 2048, out_dim = label_data.shape[1],  layer_number = 3)]

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, input_data.shape[1]])
t = tf.placeholder(tf.float32, [None, label_data.shape[1]])

def f_props(layers, x):
    for i, layer in enumerate(layers):
        x = layer.f_prop(x)
        if(i != len(layers)-1):
            x = tf.nn.dropout(x, keep_prob)
    return x

y = f_props(layers, x)

loss = tf.reduce_mean(tf.reduce_sum((y - t)**2, 1))

lrate_p = tf.placeholder(tf.float32)
mt_p = tf.placeholder(tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate = lrate_p).minimize(loss)

saver_0 = layers[0].create_saver()
saver_1 = layers[1].create_saver()
saver_2 = layers[2].create_saver()

#begin training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    # load pretrained models
    gb_model_path = 'pretrain_models/gbrbm.ckpt'
    bb1_model_path = 'pretrain_models/bbrbm_1.ckpt'
    bb2_model_path = 'pretrain_models/bbrbm_2.ckpt'
    saver_0.restore(sess, gb_model_path)
    saver_1.restore(sess, bb1_model_path)
    saver_2.restore(sess, bb2_model_path)

    n_epochs = 60
    batch_size = 128
    train_model_path = 'train_model/model.ckpt'

    n_batches = input_data.shape[1] // batch_size

    for epoch in range(n_epochs):
        lrate = 0.001

        if(epoch>10):
            lrate = 0.0005

        avg_cost = 0
        time_start = time.time()

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            _, cost = sess.run([optimizer, loss], feed_dict = {x : input_data[start:end], t : label_data[start:end], keep_prob: 0.8, lrate_p : lrate})
            avg_cost += cost / n_batches

        test_noisy_data_path = 'DataSet/test/noisy_speech/'
        test_clean_data_path = 'DataSet/test/clean_speech/'
        test_noisy_data = prepare_data(test_noisy_data_path)
        test_clean_data = prepare_data(test_clean_data_path)

        test_loss = sess.run(loss, feed_dict = {x : test_noisy_data, t : test_clean_data, keep_prob : 1})

        time_end = time.time()
        print('EPOCH: %i, average train cost: %.3f, average test cost: %.3f' % (epoch , avg_cost, test_loss))
        print('Elapsed time for one epoch is %.3f' % (time_end-time_start))
        print('\n')

    saver.save(sess, train_model_path)
    sess.close()
