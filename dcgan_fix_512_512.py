import json
import os
import time
import pathlib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2
import imageio
from tqdm import tqdm
from sklearn.utils import shuffle
from glob import glob
from ast import literal_eval

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False)

    parser.add_argument('--iterations', type=int, default=10000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch per gpu')
    parser.add_argument('--print_freq', type=int, default=100, help='The number of image_print_freqy')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')

    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')

    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of noise vector')
    parser.add_argument('--restore_model', action='store_true', default=False, help='restore_model')
    parser.add_argument('--pre_low_step', type=int, default=0, help='the latest model weights')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory name to save the checkpoints')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory name to load images')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory name to save results')

    return parser.parse_args()



def read_imgs(file_path, counts):
    imgs_list = glob(os.path.join(file_path, '*.jpg'))[:counts]
    imgs = []
    for i in tqdm(imgs_list):
        img = cv2.imread(i)[:, :, ::-1].astype(np.float32) / 255.
        imgs.append(img)
    imgs = np.array(imgs)
    
    return imgs

def build_generator(latent_dim, output_size):
    filter_num = [256, 128, 64 , 32]
    generator_input = keras.Input(shape=(latent_dim,))
    height, width = output_size
    
    x = layers.Dense(filter_num[0] * int(height//16) * int(width//16))(generator_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((int(height//16), int(width//16), filter_num[0]))(x) 
    
    #### 32*32*256

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(filter_num[0],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)  
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(filter_num[0],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)

    #### 64*64*128

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(filter_num[1],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)   
    x = layers.Conv2DTranspose(filter_num[1],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x) 
    
    #### 128*128*64

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(filter_num[2],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)   
    x = layers.Conv2DTranspose(filter_num[2],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x) 
    
    #### 256*256*32

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2DTranspose(filter_num[3],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)   
    x = layers.Conv2DTranspose(filter_num[3],(3,3),strides=(1,1),padding='same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x) 
    x = layers.Conv2DTranspose(3,(1,1),strides=(1,1),padding='same',activation='linear', kernel_initializer = 'he_normal')(x)
    
    #### 512*512*3
    
    return keras.models.Model(generator_input,x)

def build_discriminator(input_size):
    height, width, channels = input_size
    filter_num = [32,64,128,256]
    
    discriminator_input = layers.Input(shape=(height, width, channels))
    
    x = layers.Conv2D(filter_num[0], 3, padding = 'same', kernel_initializer = 'he_normal')(discriminator_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filter_num[0], 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filter_num[0], 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)
    
    #### 256*256*32

    x = layers.Conv2D(filter_num[1], 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filter_num[1], 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)

    #### 128*128*64
    
    x = layers.Conv2D(filter_num[2], 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filter_num[2], 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)
    
    #### 64*64*128
    
    x = layers.Conv2D(filter_num[3], 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filter_num[3], 3, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)
    
    #### 32*32*256

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return keras.models.Model(discriminator_input, x)

def build_GAN(G, D):
    D.trainable = False
    gan_input = G.input
    gan_output = D(G(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    return gan

def save_model(root_folder_path, model_dict):
    model_path = '{}/model/'.format(root_folder_path)
    os.makedirs(model_path, exist_ok=True)

    for key,model in model_dict.items():
        model_json = model.to_json()
        
        with open(model_path + '{}.json'.format(key), 'w') as json_file:
            json_file.write(model_json)

def train(checkpoint_dir, imgs, iterations, bs, 
          is_load_weight=False, pre_low_step=0, print_freq=100, save_freq=1000):
    start_time_all = time.time()
    
    # build net work
    latent_dim = 200
    height_1, width_1 = 512, 512

    G1 = build_generator(latent_dim, (height_1, width_1))
    D1 = build_discriminator((height_1, width_1, 3))
    GAN1 = build_GAN(G1, D1)
    
    if is_load_weight:
        model_path = './{}/model/'.format(checkpoint_dir)
        weight_path = './{}/weight/record/'.format(checkpoint_dir)
        
        with open(model_path+'G1.json', 'r') as json_file:
            temp = json_file.read()
            G1 = model_from_json(temp)
            G1.load_weights(weight_path+'g1_{}.h5'.format(pre_low_step))
    
        with open(model_path+'D1.json', 'r') as json_file:
            temp = json_file.read()
            D1 = model_from_json(temp)
            D1.load_weights(weight_path+'d1_{}.h5'.format(pre_low_step))

    optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    D1.compile(loss='binary_crossentropy', optimizer=optimizer)
    GAN1 = build_GAN(G1, D1)
    optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    GAN1.compile(loss='binary_crossentropy', optimizer=optimizer)

    model_dict = {'G1':G1,'D1':D1,'GAN1':GAN1}
    save_model(checkpoint_dir, model_dict)
    
    # create folder
    os.makedirs('{}/result_image/'.format(checkpoint_dir), exist_ok=True)
    os.makedirs('{}/weight/latest/'.format(checkpoint_dir), exist_ok=True)
    os.makedirs('{}/weight/record/'.format(checkpoint_dir), exist_ok=True)
    save_dir = '{}/result_image/'.format(checkpoint_dir)
    weight_path = '{}/weight/latest/'.format(checkpoint_dir)
    weight_record_path = '{}/weight/record/'.format(checkpoint_dir)

    # start training loop
    start = 0
    start_time = time.time()

    low_iteration = 600
    high_iteration = 1000
    pre_high_step = 0
    batch_size = bs
    batch_num = len(imgs) // batch_size

    imgs_temp = imgs[:batch_size * batch_num]

    for step in range(iterations):
        imgs_temp = shuffle(imgs_temp)

        for low_step in range(batch_num):
            real_images = imgs_temp[low_step * batch_size:(low_step + 1) * batch_size]
            real_images = (real_images-0.5)*2
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

            generated_images = G1.predict(random_latent_vectors)

            labels = np.concatenate([np.zeros((batch_size, 1)),
                                 np.ones((batch_size, 1))])

            labels_real = 0.9 * np.ones((batch_size, 1)) 
            labels_fake = np.zeros((batch_size, 1)) 

            d_loss_real = D1.train_on_batch(real_images, labels_real)
            d_loss_fake = D1.train_on_batch(generated_images, labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

            misleading_targets = np.ones((batch_size, 1))

            g_loss = GAN1.train_on_batch(random_latent_vectors, misleading_targets)

            if low_step % print_freq == 0:
                # save model weights
                G1.save_weights(weight_path+'g1.h5')
                D1.save_weights(weight_path+'d1.h5')
                
                step_indicator = step*low_iteration+low_step+pre_low_step
                
                if step_indicator % save_freq == 0:
                    G1.save_weights(weight_record_path+'g1_{}.h5'.format(step_indicator))
                    D1.save_weights(weight_record_path+'d1_{}.h5'.format(step_indicator))

                # print metrics
                print('low resolution, discriminator loss at step %s: %s' % (step_indicator, d_loss))
                print('low resolution, adversarial loss at step %s: %s' % (step_indicator, g_loss))
                display_grid = np.zeros((4*height_1,width_1,3))

                for j in range(4):
                    display_grid[j*height_1:(j+1)*height_1,0:width_1,:] = generated_images[j]

                img = image.array_to_img((display_grid[:,:,::-1]*127.5)+127.5, scale=False)
                img.save(os.path.join(save_dir, 'low_generated_' + str(step*low_iteration+low_step+pre_low_step) + '.png'))
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
def test(result_dir, checkpoint_dir, bs=4):
    os.makedirs(result_dir, exist_ok=True)
    
    # build net work
    latent_dim = 200
    height_1, width_1 = 512, 512

    G1 = build_generator(latent_dim, (height_1, width_1))
    
    model_path = '{}/model/'.format(checkpoint_dir)
    weight_path = '{}/weight/latest/'.format(checkpoint_dir)
        
    with open(model_path+'G1.json', 'r') as json_file:
        G1 = model_from_json(json_file.read())
    G1.load_weights(weight_path + 'g1.h5')


    random_latent_vectors = np.random.normal(size=(bs, latent_dim))
    generated_images = G1.predict(random_latent_vectors)
    
    display_grid = np.empty((bs * height_1, width_1, 3))
    for j in range(bs):
        display_grid[j * height_1:(j + 1) * height_1, 0:width_1, :] = generated_images[j]
        img = image.array_to_img((display_grid[:,:,::-1] * 127.5) + 127.5, scale=False)
        img.save(os.path.join(result_dir, 'result.png'))
                
if __name__ == '__main__':
    args = arg_parse()
    
    # load image
    imgs_length = 4000
    imgs = read_imgs(args.data_dir, imgs_length)
    
    # train
    if args.train:
        train(args.checkpoint_dir, imgs, 
              iterations=args.iterations, bs=args.batch_size,
              is_load_weight=args.restore_model,
              pre_low_step=args.pre_low_step,
              print_freq=args.print_freq,
              save_freq=args.save_freq)
        
    else:
        test(args.result_dir, args.checkpoint_dir, bs=args.batch_size)
