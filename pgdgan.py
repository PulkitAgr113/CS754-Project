import os
import numpy as np
from argparse import ArgumentParser
import celebA_estimators
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()
m = 1000
n = 64 * 64 * 3
A = np.random.normal (0, m, (m,n))

T = 10
T_in = 100
lr = 0.5
lr_in = 0.1

def crop(x, crop):
    h, w = x.shape[:2]
    j = int(round((h - crop)/2.))
    i = int(round((w - crop)/2.))
    return np.reshape(x[j:j+crop, i:i+crop], [crop, crop, 3])

def transform(image):
    cropped_image = crop(image, 64)
    return np.array(cropped_image)/127.5 - 1.


parser = ArgumentParser()

params, unknown = parser.parse_known_args()
params.batch_size = 64
params.n_input = n
params.mloss1_weight = 0.0
params.mloss2_weight = 1.0
params.zprior_weight = 0.001
params.dloss1_weight = 0.0
params.dloss2_weight = 0.0
params.num_random_restarts = 1
params.max_update_iter = T_in
params.pretrained_model_dir = './models/celebA_64_64/'
params.decay_lr = True
params.learning_rate = lr_in
params.optimizer_type = 'adam'


x_batch = np.zeros((64, n))
k = 0
for image in sorted(os.listdir('data/celebAtest')):
  x = transform(plt.imread('data/celebAtest/' + image).astype(np.float))
  x = np.reshape(x, (1,-1))
  x_batch[k] = x
  print(image)
  k += 1

new_img = np.reshape(x_batch[0],(64,64,3))*127.5+127.5
plt.imsave("General Kenobi", new_img)
y = np.matmul(x_batch, A.T)
x_main = 0.0 * x_batch
z = np.random.randn(64, 100)

for t in range(T):
  x_est = x_main + lr * np.matmul(y - np.matmul (x_main, A.T), A)
  estimator = celebA_estimators.dcgan_estimator(params)
  x_main, z = estimator (x_est, z, params)

new_img = np.reshape(x_main[0],(64,64,3))*127.5+127.5
plt.imsave("Hello there", new_img)
