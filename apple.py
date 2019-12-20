import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorboard
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp

from tensorflow import keras
from pandas.io.parsers import read_csv


tf.compat.v1.disable_eager_execution()


tfd = tfp.distributions

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(1,kernel_initializer='glorot_uniform'),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    ])


model = tf.global_variables_initializer();
#데이터 파일 가져오기
data = read_csv('apple data.csv', sep=',')

xy = np.array(data, dtype=np.float32)

x_data = xy[:, 1:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#학습하기
for step in range(10000):

    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 500 == 0:
        print("*", step, " 손실비용: ", cost_)
        print("사과 가격: ", hypo_[0])
#학습된 모델 저장하기
    saver = tf.train.Saver()

    save_path = saver.save(sess, "./saved.cpkt")










