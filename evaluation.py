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


#modelpath = "/home/pdp_model/revision/d1_coeff_generator_10000e_tgan_pdap_latent=128_r_scale_180_twc_eoa2_revision.h"
#modelpath =  "/home/pdp_model/revision/d1_coeff_generator_10000e_tgan_pdap_latent=128_r_scale_180_twc_eoa2_revision_maxDelay=400.h"
#modelpath = "/home/pdp_model/revision/d1_real_21_coeff_generator_10000e_tgan_l2norm_scale180_twc_eoa2_revision_maxDelay=400_0121.h/"
#outputpath = "revision/new_augmentation_maxDelay=400_after_transfer.mat"
#plt.plot(np.transpose(data1['epoch'][:]),np.transpose(data1['d_loss'][:]),'b',label='d_loss')
modelpath = "/home/pdp_model/revision/env2/d1_real_21_coeff_generator_10000e_tgan_l2norm_scale180_twc_eoa2_01"+str(args.samples)+"times="+str(args.times)+".h"
outputpath = 'revision/env2_average/'+str(args.samples)+"_"+str(args.times)+'.mat'


#generator_C = tf.keras.models.load_model('coeff_generator_10000_wcgan_pdap_latent=128.h')
##---generator_C = tf.keras.models.load_model('/root/hzd/pdp_model/bench_coeff_generator_10000e_tgan_pdap_latent=128_r.h')
#generator_C = tf.keras.models.load_model('/root/hzd/pdp_model/epoch/d4_10000e_epoch_1000_gloss_0.5799226760864258.h')
generator_C = tf.keras.models.load_model(modelpath)
#d1_real_21_coeff_generator_10000e_tgan_l2norm_scale180_twc_eoa2.h
#d1_coeff_generator_10000e_tgan_pdap_latent=128_r_scale_180_twc_eoa2.h
#generator_C = tf.keras.models.load_model('/home/pdp_model/d1_coeff_generator_10000e_tgan_pdap_latent=128_r_scale_180_twc_neoa.h')
#coeff_generator_10000_tgan_pdap_latent=128.h
#generator_C = tf.keras.models.load_model('/root/hzd/pdp_model/epoch/pad_model_epoch_400_gloss_1.499739170074463.h')
#data_condition = h5py.File('input_coeff/coeff_rx_position_50000_minus_path20_pad.mat','r')
random_latent_vectors = tf.random.normal(shape=(10000, latent))
data_condition = h5py.File('input_coeff/coeff_rx_position_10000.mat','r')
#random_latent_vectors = tf.random.normal(shape=(21, latent))
#data_condition = h5py.File('input_coeff/coeff_rx_position_21.mat','r')
 #data_train = h5py.File('input/pdp_10000_401_new_30m.mat', 'r')
#data_train = np.transpose(data_train['H'][:])#Channel
data_condition = np.transpose(data_condition['dis'][:])
#data_condition = np.tile(data_condition,(100,1))
#data_condition = data_condition[40000:,:]
l = time.time()
fake_channel = generator_C([random_latent_vectors,data_condition], training=False)
l2 = time.time()
print(l2-l)
channel_plot = fake_channel.numpy()
channel_plot = np.reshape(channel_plot, (10000,15,4))
"""
channel_num = np.arange(401)
plt.figure()
plt.plot(channel_num,channel_plot[2,:])
plt.savefig('simulated_distribution2.jpg')
print(type(channel_plot))
print(type(channel_num))
"""


savemat(outputpath, {'channel': channel_plot})



"""
x = np.arange(36)
y = np.arange(400)
X,Y = np.meshgrid(x,y)
Z = np.reshape(np.squeeze(channel_plot[10,:]),(400,36))
fig, ax = plt.subplots()
cs = ax.contourf(X,Y, Z,cmap=plt.get_cmap('Spectral'))
cbar = fig.colorbar(cs)
plt.show()
plt.savefig('simulated_distribution.jpg')
"""
