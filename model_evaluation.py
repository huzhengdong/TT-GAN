import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import argparse
import matplotlib.pyplot as plt
#from scipy.stats import ks_2samp
import scipy.stats
from scipy.io import savemat
import time
latent = 32

parser = argparse.ArgumentParser()
parser.add_argument('--samples','-s',help='Number of samples',default=10)
parser.add_argument('--times','-t',help='Number of times',default=1)
args = parser.parse_args()



modelpath = "/home/pdp_model/revision/env2/d1_real_21_coeff_generator_10000e_tgan_l2norm_scale180_twc_eoa2_01"+str(args.samples)+"times="+str(args.times)+".h"
outputpath = 'revision/env2_average/'+str(args.samples)+"_"+str(args.times)+'.mat'



generator_C = tf.keras.models.load_model(modelpath)
random_latent_vectors = tf.random.normal(shape=(10000, latent))
data_condition = h5py.File('input_coeff/coeff_rx_position_10000.mat','r')
data_condition = np.transpose(data_condition['dis'][:])
l = time.time()
fake_channel = generator_C([random_latent_vectors,data_condition], training=False)
l2 = time.time()
print(l2-l)
channel_plot = fake_channel.numpy()
channel_plot = np.reshape(channel_plot, (10000,15,4))
savemat(outputpath, {'channel': channel_plot})




