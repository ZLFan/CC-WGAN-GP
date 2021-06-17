import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()

os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# generate vanilla eeg during training
def get_noise():
    PATH = '/workspace/CC-WGAN-GP/DATA/datanoise.mat'
    data = sio.loadmat(PATH)
    train_noise = data['eeg_noise']
    train_noise = np.array(train_noise)
    # print(train_noise.shape)
    all_noise = np.zeros((train_noise.shape[0] * 32, 231))
    for i in range(0, train_noise.shape[0]):
        for j in range(0, 32):
            all_noise[i * j, :] = train_noise[i, j, :]
    all_noise = all_noise[:, 0:231]
    # all_noise = np.squeeze(all_noise, axis=2)
    all_noise = mm.fit_transform(all_noise)
    all_noise = np.expand_dims(all_noise, axis=2)
    print(all_noise.shape)

    return all_noise
def generate_eeg(generator_model, dir, epoch, freq):

    if epoch%freq==0:
        dir='/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/'
        # eeg_gen = generator_model.predict(np.random.normal(0, 1, (100, 120)))
        #脑电噪声
        idx = np.random.randint(0, 121056, 1252)
        # idx = np.random.randint(0, 121056, 100)
        X_noise = get_noise()
        noise = X_noise[idx,0:120,:]
        # print("noise:",noise.shape) 
        noise = np.squeeze(noise, axis=2).astype(np.float32)
        eeg_gen = generator_model.predict(noise)
        np.save(str(dir) + '/eeg_generated/' + str(epoch) + '.npy', eeg_gen) #comment this, if you don't want to save samples
        # generator_model.save(str(dir) + '/saved_model/' + 'generator_' + str(epoch) + '.h5')
        # uncomment if you want to save model

# generate conditional eeg during training (used in one channel GAN)
def generate_condi_eeg(generator_model, dir, epoch, freq):
    dir='/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/'
    if epoch%freq==0:
        if not os.path.exists('/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/eeg_generated/target_/'):
            os.makedirs('/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/eeg_generated/target_/')
        if not os.path.exists('/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/eeg_generated/nontarget_/'):
            os.makedirs('/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/eeg_generated/nontarget_/')
        # randomly input to the generator
        # noise = np.random.normal(0, 1, (100, 120)).astype(np.float32)
        # nontarget_labels = np.random.randint(0, 1, 100).reshape(-1, 1)
        # target_labels = np.random.randint(1, 2, 100).reshape(-1, 1)
        #noise = np.random.normal(0, 1, (1252, 120)).astype(np.float32)

        #脑电噪声
        idx = np.random.randint(0, 121056, 1252)
        # idx = np.random.randint(0, 121056, 100)
        X_noise = get_noise()
        noise = X_noise[idx,0:120,:]
        # print("noise:",noise.shape) 
        noise = np.squeeze(noise, axis=2).astype(np.float32)

        # nontarget_labels = np.random.randint(0, 1, 100).reshape(-1, 1)
        # target_labels = np.random.randint(1, 2, 100).reshape(-1, 1)
        nontarget_labels = np.random.randint(0, 1, 1252).reshape(-1, 1)
        target_labels = np.random.randint(1, 2, 1252).reshape(-1, 1)
        # generate target & non-target samples
        generated_target = generator_model.predict([noise, target_labels.astype('float32')])
        generated_nontarget = generator_model.predict([noise, nontarget_labels.astype('float32')])
        # comment below, if you don't want to save samples
        # np.save(str(dir) + '/eeg_generated/target_' + str(epoch) + '.npy', generated_target)
        print("generated_target.shape: ",generated_target.shape)
        print("generated_nontarget.shape: ",generated_nontarget.shape)
        sio.savemat('/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/eeg_generated/target_/%d.mat' % (epoch), {'eeg': generated_target})
        # np.save(str(dir) + '/eeg_generated/nontarget_' + str(epoch) + '.npy', generated_nontarget)
        sio.savemat('/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/eeg_generated/nontarget_/%d.mat' % (epoch), {'eeg': generated_nontarget})
        # comment below, if you don't want to save model
        # generator_model.save(str(dir) + '/saved_model/' + 'generator_' + str(epoch) + '.h5')

# save trained CC-GAN models, generated target & nontarget for best classifier AUC
def save_CC_GAN(generator_model, discriminator_model, dir):
        # randomly input to the generator
        dir='/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/'
        noise = np.random.normal(0, 1, (784, 120)).astype(np.float32)
        nontarget_labels = np.random.randint(0, 1, 784).reshape(-1, 1)
        target_labels = np.random.randint(1, 2, 784).reshape(-1, 1)
        # generate target & non-target samples
        generated_target = generator_model.predict([noise, target_labels.astype('float32')])
        generated_nontarget = generator_model.predict([noise, nontarget_labels.astype('float32')])
        # Save target and nontarget samples as numpy data
        np.save(str(dir) + '/saved_model/target.npy', generated_target)
        np.save(str(dir) + '/saved_model/nontarget.npy', generated_nontarget)
        # Save Generator and Discriminator models
        generator_model.save(str(dir) + '/saved_model/' + 'CC_GAN_generator.h5')
        discriminator_model.save(str(dir) + '/saved_model/' + 'CC_GAN_discriminator.h5')

# classifier test AUC during training
def plot_AUC(AUC, dir):
    dir='/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/'
    plt.plot(AUC, label="Classifier_AUC")
    plt.title('Classifier Performace')
    plt.savefig(str(dir) + '/evaluation/' + 'Classifier_AUC.png')
    plt.legend()
    plt.show()

# if you want to save weights and model architecture separate
def save(model, dir, model_name):
    dir='/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/'
    model_path = str(dir) + '/saved_model/' + str(model_name) +'.json'
    weights_path = str(dir) + '/saved_model/' + str(model_name)+'_weights.h5'
    json_string = model.to_json()

    open(model_path, 'w').write(json_string)
    model.save_weights(weights_path)

# WGAN loss plots
def plot_losses(discrminator_loss, generator_loss, dir):
    dir='/workspace/CC-WGAN-GP/RESULT/add-eeg-noise/'
    plt.plot([i for i in range(discrminator_loss.shape[0])], 1 - np.array(discrminator_loss), label="D_loss")
    plt.plot(np.arange(0, discrminator_loss.shape[0], (discrminator_loss.shape[0]/generator_loss.shape[0])),
             1 - np.array(generator_loss), label="G_loss")
    plt.title('Loss-curve')
    plt.legend()
    plt.savefig(dir + '/evaluation/'+'loss_curve.png')
    plt.show()
