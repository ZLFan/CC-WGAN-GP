# Author: Sharaj Panwar, MSEE; Research Fellow at Brain Computer Interface Lab./& Open Cloud Institute at UTSA, San Antonio, Texas
# Acknowledgement: this code takes basic WGAN framework from keras implementation "https://github.com/keras-team/
# keras-contrib/blob/master/examples/improved_wgan.py" of the improved WGAN described in https://arxiv.org/abs/
from __future__ import print_function, division
import tensorflow as tf
import keras
import os
import scipy.io as sio
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
import pandas as pd
import seaborn as sns
# importing custom modules created for GAN training
from data_loader import data_import_ch1
from out_put_module import generate_condi_eeg, plot_losses
from wgan_gp_loss import wasserstein_loss, gradient_penalty_loss
from arch_one_channel import eeg_generator, eeg_discriminator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

sns.set()
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

batch_size = 64
training_ratio = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GP_Weight = 10  # As per the paper
sample_freq = 1 # frequency at which you want to generate image or save generator models
Number_epochs = 100


# # uploading target data for the given subject
# Target = np.load('/home-new/ijz121/PycharmProjects/gan_projects/gan_rsvp_project/'
#                  'rsvp_data/data_rsvp_preprocessed/poz_data_preprocessed/s1_r2_poz_T.npy')
# # uploading nontarget data for the given subject
# nonTarget = np.load('/home-new/ijz121/PycharmProjects/gan_projects/gan_rsvp_project/'
#                  'rsvp_data/data_rsvp_preprocessed/poz_data_preprocessed/s1_r2_poz_nT.npy')
# # creating labels
# y_Target = np.ones(Target.shape[0]); y_nonTarget = np.zeros(nonTarget.shape[0])
# X= np.concatenate((Target, nonTarget), axis=0); y = np.concatenate((y_Target, y_nonTarget), axis=0)
data1=sio.loadmat('/workspace/CC-WGAN-GP/DATA/ta.mat')
data2=sio.loadmat('/workspace/CC-WGAN-GP/DATA/tb.mat')
data1=data1['ta']
data1=np.array(data1)
# data1.reshape(data1,(1252,231,1))
t1 = np.zeros((1252,231,1))
t1[0:1251,0:230,0]=data1[0:1251,0:230]
data2=data2['tb']
data2=np.array(data2)
# data2.reshape(data2,(1252,231,1))
t2 = np.zeros((1252,231,1))
t2[0:1251,0:230,0]=data2[0:1251,0:230]
print("t1.shape: ",t1.shape)
print("t2.shape: ",t2.shape)
# PATH = 'F:/AI/CODE/session1_epoch/'
# data = sio.loadmat(PATH+ 'subject_'+'1.mat')
# train_eeg = data['data']
# train_eeg = np.array(train_eeg)
# print(train_eeg.shape)
# all_data = np.zeros((train_eeg.shape[2],231,32))
#
# for i in range(0,32):
#     for j in range(0,train_eeg.shape[2]):
#         all_data[j,:,i]=train_eeg[i,:,j]
# subject = 35
# for i in range(2,subject+1):
#     data = sio.loadmat(PATH+ 'subject_'+str(i)+'.mat')
#     train_eeg = data['data']
#     #train_eeg = np.array(train_eeg)
#     newdata = np.zeros((train_eeg.shape[2],231,32))
#     for i in range(0, 32):
#         for j in range(0,train_eeg.shape[2]):
#             newdata[j,:,i]=train_eeg[i,:,j]
#     all_data = np.r_[all_data,newdata]
# all_data =  all_data [:,0:231 ,12:13]
# print(all_data.shape)
# # all_data  = np.squeeze(all_data , axis=2)
# # all_data = mm.fit_transform(all_data)
# # all_data= np.expand_dims(all_data, axis=2)
# x= all_data
Target=t1
nonTarget=t2
y_Target = np.ones(Target.shape[0]); y_nonTarget = np.zeros(nonTarget.shape[0])
X= np.concatenate((Target, nonTarget), axis=0); y = np.concatenate((y_Target, y_nonTarget), axis=0)
# Shuffle and reshape the data to fit in model
X_train, y_train = data_import_ch1(X, y)

# directories to store the results
output_dir = '/workspace/CC-WGAN-GP/RESULT/newdata/'
# Run tag will dynamically create the sub-directories to store results
Run_tag = 'one_channel_gan_' + str(Number_epochs) + '_epoch/'
#  sub-directories to store results
dir = os.path.join(output_dir, Run_tag)
print (Run_tag)
if not os.path.exists(dir):
    os.makedirs(dir)
    print("dir",dir)
    os.makedirs(dir + '/eeg_generated')
    os.makedirs(dir + '/evaluation')
    os.makedirs(dir + '/saved_model')
# else:
    # raise Exception('Excuse me Boss!! Please change the Run Tag to avoid rewriting files ;)')

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

# create generator and discriminator models imported from arch_one_channel
generator = eeg_generator()
discriminator = eeg_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
noise = Input(shape=(120,))
label = Input(shape=(1,), dtype='int32')

generator_input = ([noise, label])
generator_layers = generator(generator_input)
print("generator_layers",generator_layers)
print("label",label)
discriminator_layers_for_generator = discriminator([generator_layers, label])
generator_model = Model(inputs=([noise, label]), outputs=[discriminator_layers_for_generator])
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(120,))
label = Input(shape=(1,), dtype='int32')

generated_samples_for_discriminator = generator([generator_input_for_discriminator, label])
discriminator_output_from_generator = discriminator([generated_samples_for_discriminator, label])
discriminator_output_from_real_samples = discriminator([real_samples, label])

averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
averaged_samples_out = discriminator([averaged_samples, label])

partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GP_Weight)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error


discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator, label],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

positive_y = np.ones((batch_size, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

epochs = [];
wgan_d_itr =[];wgan_g_itr=[]; #discriminator and generator losses for each iteration in training ratio
wgan_d_ep=[];wgan_g_ep=[]; #discriminator and generator losses for each epoch averaging over training ratio

# def get_noise(self):
#     PATH = 'datanoise.mat'
#     data = sio.loadmat(PATH)
#     train_noise = data['eeg_noise']
#     train_noise = np.array(train_noise)
#     # print(train_noise.shape)
#     all_noise = np.zeros((train_noise.shape[0] * 32, 231))
#     for i in range(0, train_noise.shape[0]):
#         for j in range(0, 32):
#             all_noise[i * j, :] = train_noise[i, j, :]
#     all_noise = all_noise[:, 0:231]
#     # all_noise = np.squeeze(all_noise, axis=2)
#     all_noise = preprocessing.MaxAbsScaler().fit_transform(all_noise)
#     all_noise = np.expand_dims(all_noise, axis=2)
#     print(all_noise.shape)

#     return all_noise

for epoch in range(Number_epochs):
    np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // batch_size))
    discriminator_loss = []
    generator_loss = []

    minibatches_size = batch_size * training_ratio
    for i in range(int(X_train.shape[0] // (batch_size * training_ratio))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        y_value_minibatches = y_train[i * minibatches_size:(i + 1) * minibatches_size]

        for j in range(training_ratio):
            image_batch = discriminator_minibatches[j * batch_size:(j + 1) * batch_size]
            y_value = y_value_minibatches[j * batch_size:(j + 1) * batch_size]
            #????????????
            # idx = np.random.randint(0, batch_size, 120)
            # noise = X_noise[idx]
            # noise = np.squeeze(noise, axis=2).astype(np.float32)

            # noise = np.random.normal(-1, 1, (batch_size, 120)).astype(np.float32)
            noise = np.random.normal(0, 1, (batch_size, 120)).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch,noise, y_value],
                                                                         [positive_y, negative_y, dummy_y]))

        sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)

        # generator_loss.append(generator_model.train_on_batch([np.random.normal(-1, 1, (batch_size, 120)), sampled_labels], positive_y))
        generator_loss.append(generator_model.train_on_batch([np.random.normal(0, 1, (batch_size, 120)), sampled_labels], positive_y))

    epochs.append(epoch)
    wgan_d_itr.append(np.array(discriminator_loss)[:, 0])
    wgan_g_itr.append(np.array(generator_loss))

    wgan_d_ep.append(np.mean(discriminator_loss, axis=0))
    wgan_g_ep.append(np.mean(np.array(generator_loss), axis=0))

    generate_condi_eeg(generator, dir, epoch, sample_freq)

# saving 'losses, average over training ratio' to csv file
validation_matrix = pd.DataFrame({'epoch': epochs, 'Discriminator_loss': wgan_d_ep, 'Generator_loss': wgan_g_ep})
validation_matrix.to_csv(str(dir) + '/evaluation/'+ 'validation_matrix.csv')

# plotting losses per iteration
wgan_d_itr = np.array(wgan_d_itr).flatten()
wgan_g_itr = np.array(wgan_g_itr).flatten()
plot_losses(wgan_d_itr, wgan_g_itr, dir)