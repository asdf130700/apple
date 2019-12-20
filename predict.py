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

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

saver = tf.train.Saver()
model = tf.global_variables_initializer()


#변수 입력받기

avgtem = float(input('평균 기온: '))
mintem = float(input('최저 기온: '))
maxtem = float(input('최고 기온: '))
rain = float(input('강수량: '))

with tf.Session() as session:
    session.run(model)
   # 저장된 파일 불러오기

    save_path = "./saved.cpkt"
    saver.restore(session, save_path)

    data = ((avgtem, mintem, maxtem, rain), )
    arr = np.array(data, dtype=np.float32)
    x_data = arr[0:4]
    dict = session.run(hypothesis, feed_dict={X: x_data})
    print(dict[0])