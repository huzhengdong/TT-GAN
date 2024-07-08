#==================================================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.io as io
from keras.callbacks import ModelCheckpoint
import time
from modelDesign_coeffr import *
import os
#==================================================================================================
# GPU Configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

#==================================================================================================
# Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--epochs','-e',help='Number of epochs',default=10)
args = parser.parse_args()
latent = 32
length = 15*4
batch_S = 256 #16

#==================================================================================================
# Function definition

def data_processing():
    print('Data Loading...')
    #data_train = h5py.File('input_coeff/coeff_10000_d1_scale_180.mat', 'r')#Channel_set
    #data_condition = h5py.File('input_coeff/coeff_rx_position_10000.mat','r')
    data_train = h5py.File('input_coeff/coeff_10000_path15_new_config_eoa2.mat', 'r')#Channel_set
    data_condition = h5py.File('input_coeff/coeff_rx_position_path15_new_config_eoa2.mat','r')
    
    #data_train = h5py.File('input/pdp_10000_401_new_30m.mat', 'r')
    data_train = np.transpose(data_train['H'][:])#Channel
    data_condition = np.transpose(data_condition['dis'][:])
    #data_train = data_train[:40000,:,:]
    #data_condition = data_condition[:40000,:]
    numSample = np.size(data_train,0)
    #numSample = 1000
    index = np.random.shuffle(np.arange(numSample))
    train_channel = np.squeeze(data_train[index],0)
    train_condition = np.squeeze(data_condition[index],0)
    #train_channel = np.transpose(train_channel,(0,2,1))
    print(train_condition.shape)
    print(train_channel.shape)
    train_channel = np.reshape(train_channel,(numSample,-1))
    print(train_channel.shape)  #[None,shape]
    return train_channel, train_condition

def discriminator_loss(real_img, fake_img):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_img), logits=real_img)  # label=1
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_img), logits=fake_img)  # label=0
    #total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    total_disc_loss = tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)
    return total_disc_loss

                    
def generator_loss(fake_img):
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_img), logits=fake_img)
    return -tf.reduce_mean(fake_img)


#==================================================================================================

def oDiscriminator():
    img_input = layers.Input(shape=length)
    label = layers.Input(shape=(1,))
    y = layers.Dense(16)(label)
    y = layers.LeakyReLU(alpha=0.2)(y)
    x = layers.concatenate([img_input,y])
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1)(x)
    d_model = keras.models.Model(inputs=[img_input,label], outputs=x, name="discriminator")
    return d_model
def oGenerator():
    noise = layers.Input(shape=(latent,))
    label = layers.Input(shape=(1,))
    y = layers.Dense(16)(label)
    y = layers.LeakyReLU(alpha=0.2)(y)
    x = layers.concatenate([noise,y])
    x = layers.Dense(64)(x)#4*4*64
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(length,activation='sigmoid')(x)
    g_model = keras.models.Model(inputs=[noise,label],outputs=x, name="generator")
    return g_model
#==================================================================================================





def Discriminator():
    img_input = layers.Input(shape=length)
    label = layers.Input(shape=(1,))
    #label= tf.tile(label,[1,36])
    x = layers.concatenate([img_input,label])
    #x = layers.Dense(length)(x)
    x = Decoder(x)
    #x = layers.Dense(2048)(img_input)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dense(1024)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dense(512)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dense(256)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dense(1)(x)
    d_model = keras.models.Model(inputs=[img_input,label], outputs=x, name="discriminator")
    return d_model

def Generator():
    noise = layers.Input(shape=(latent,))
    label = layers.Input(shape=(1,))
    x = layers.concatenate([noise,label])
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.BatchNormalization()(x)
    x = Encoder(x)#4*4*64
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dense(512)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dense(1024)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dense(2048)(x)
    #x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dense(length,activation='sigmoid')(x)
    g_model = keras.models.Model(inputs=[noise,label],outputs=x, name="generator")
    return g_model







class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=1,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn,loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.loss_fn = loss_fn

    def gradient_penalty(self, batch_size, real_channel, fake_channel,condition):
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_channel - real_channel
        interpolated = real_channel + alpha * diff
        diff = fake_channel - real_channel
        interpolated = real_channel + alpha * diff
        print(alpha)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated,condition], training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_channel, label = data
       # print(data)
        if isinstance(real_channel, tuple):
            real_channel = real_channel[0]
       # print(real_channel)
        batch_size = tf.shape(real_channel)[0]
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_label = tf.random.uniform(shape=(batch_size,1),minval=0,maxval=1)
            #random_label_z = tf.random.uniform(shape=(batch_size,1),minval=0,maxval=1)
            #random_label = tf.concat([random_label_xy,random_label_z],-1)
            #latent = tf.concat([random_latent_vectors,batch_target],-1)
            with tf.GradientTape() as tape:
                fake_channel = self.generator([random_latent_vectors,label], training=True)
                fake_logits = self.discriminator([fake_channel,label], training=True)
                real_logits = self.discriminator([real_channel,label], training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size,real_channel,fake_channel,label)
                d_loss = d_cost + gp*self.gp_weight
                #print(fake_channel)
                #print(real_channel)
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        #random_label_xy = tf.random.uniform(shape=(batch_size,2),minval=-1,maxval=1)
        #random_label_z = tf.random.uniform(shape=(batch_size,1),minval=0,maxval=1)
        #random_label = tf.concat([random_label_xy,random_label_z],-1)
        random_label = tf.random.uniform(shape=(batch_size,1),minval=0,maxval=1)

        with tf.GradientTape() as tape:
            generated_images = self.generator([random_latent_vectors,label], training=True)
            gen_img_logits = self.discriminator([generated_images,label], training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
            
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

    def plot(self):
        random_latent_vectors = tf.random.normal(shape=(100, self.latent_dim))
        latent = tf.concat(random_latent_vectors,-1)
        fake_channel = self.generator(random_latent_vectors, training=False)
        channel_plot = fake_channel.numpy()
        #channel_num = np.arange(401)
        x = np.arange(401)
        y = np.arange(36)
        X,Y = np.meshgrid(x,y)
        Z = np.reshape(np.squeeze(channel_plot[10,:]),(36,401))
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Z, cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.show()
        plt.savefig('simulated_distribution.jpg')
        return channel_plot

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if( epoch % 1000 == 0):
      #print("\nReached 60% accuracy so cancelling training!")
      #self.model.stop_training = True
        self.model.generator.save(os.path.join('epoch/', 'd1_10000e_epoch_{}_gloss_{}_scale_180_eoa2.h'.format(epoch,logs['g_loss'])))
        self.model.discriminator.save(os.path.join('epoch/', 'd1_10000e_epoch_discriminator_epoch_{}_dloss_{}_scale_180_eoa2.h'.format(epoch,logs['d_loss'])))

#==================================================================================================


#channel,condition = data_processing()
g_model = Generator()
#g_model = tf.keras.models.load_model('generator_50000_wgan_pdap_latent=128_v2.h')
g_model.summary()

d_model = Discriminator()
#d_model = tf.keras.models.load_model('discriminator_50000_wgan_pdap_v2.h')
d_model.summary()

channel,condition = data_processing()


"""
for layer in d_model.layers:
        print(layer.name, ' is trainable? ', layer.trainable)
d_model.layers[1].trainable = False
d_model.layers[3].trainable = False
d_model.layers[5].trainable = False
for layer in d_model.layers:
    print(layer.name,'is trainable?', layer.trainable)
"""


#lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_steps=1000,decay_rate=0.5)
generator_optimizer = keras.optimizers.SGD(learning_rate=0.0002) #%0.0001
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002) #%0.0001


wgan = WGAN(discriminator=d_model,generator=g_model,latent_dim=latent,discriminator_extra_steps=2,)
wgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer,g_loss_fn=generator_loss,d_loss_fn=discriminator_loss,loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),)

#filepath = 'best_gan.h'
#checkpoint = ModelCheckpoint(filepath, monitor='d_loss', verbose=1,save_best_only=True,mode='max',period=2) 
checkpoint = myCallback()
callbacks_list = [checkpoint]
t1 = time.time()
history=wgan.fit(channel,condition,batch_size=batch_S, epochs=int(args.epochs),callbacks=callbacks_list)
t2 = time.time()
print(t2-t1)
#wgan.save('wgan_10000.h')

wgan.generator.save('d1_coeff_generator_10000e_tgan_pdap_latent=128_r_scale_180_twc_eoa2.h')
wgan.discriminator.save('d1_coeff_discriminator_10000e_tgan_pdap_r_scale_180_twc_eoa2.h')


#channel_save = wgan.plot()
print(history.history.keys())
epochs=range(len(history.history['d_loss']))
io.savemat('loss_10000e_pdap_scale_180_twc_eoa2.mat',mdict={'epoch':epochs,'d_loss':history.history['d_loss'],'g_loss':history.history['g_loss']})
#print(history.history['d_loss'])
plt.figure()
plt.plot(epochs,history.history['d_loss'],'b',label='d_loss')
plt.plot(epochs,history.history['g_loss'],'r',label='g_loss')
plt.title('Generator and Discriminator loss')
plt.legend()
plt.savefig('loss_d1_10000_scale_180_eoa2.jpg')

