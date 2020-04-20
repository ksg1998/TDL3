import tensorflow as tf
import numpy as np
import scipy
import imageio
import os
import shutil
from PIL import Image
import time
import random


from layers import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width


batch_size = 1
pool_size = 50
ngf = 32
ndf = 64

def resnet_builder(input, dim, name="resnet"):

    with tf.variable_scope(name):

        resnet_op = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        resnet_op = general_conv2d(resnet_op, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        resnet_op = tf.pad(resnet_op, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        resnet_op = general_conv2d(resnet_op, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)
        
        return tf.nn.relu(resnet_op + input)

def resnet_generator_6(input, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(input,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02,name="c1")
        o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
        o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

        o_r1 = resnet_builder(o_c3, ngf*4, "r1")
        o_r2 = resnet_builder(o_r1, ngf*4, "r2")
        o_r3 = resnet_builder(o_r2, ngf*4, "r3")
        o_r4 = resnet_builder(o_r3, ngf*4, "r4")
        o_r5 = resnet_builder(o_r4, ngf*4, "r5")
        o_r6 = resnet_builder(o_r5, ngf*4, "r6")

        o_c4 = general_deconv2d(o_r6, [batch_size,64,64,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
        o_c5 = general_deconv2d(o_c4, [batch_size,128,128,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
        o_c5_pad = tf.pad(o_c5,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv2d(o_c5_pad, img_layer, f, f, 1, 1, 0.02,"VALID","c6",do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6,"t1")


        return out_gen

def resnet_generator_9(input, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(input,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02,name="c1")
        o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
        o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

        o_r1 = resnet_builder(o_c3, ngf*4, "r1")
        o_r2 = resnet_builder(o_r1, ngf*4, "r2")
        o_r3 = resnet_builder(o_r2, ngf*4, "r3")
        o_r4 = resnet_builder(o_r3, ngf*4, "r4")
        o_r5 = resnet_builder(o_r4, ngf*4, "r5")
        o_r6 = resnet_builder(o_r5, ngf*4, "r6")
        o_r7 = resnet_builder(o_r6, ngf*4, "r7")
        o_r8 = resnet_builder(o_r7, ngf*4, "r8")
        o_r9 = resnet_builder(o_r8, ngf*4, "r9")

        o_c4 = general_deconv2d(o_r9, [batch_size,128,128,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
        o_c5 = general_deconv2d(o_c4, [batch_size,256,256,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
        o_c6 = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02,"SAME","c6",do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6,"t1")


        return out_gen

def discriminator_builder(input, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5

def patch_discriminator(input, name="discriminator"):

    with tf.variable_scope(name):
        f= 4

        patch_input = tf.random_crop(input,[1,70,70,3])
        o_c1 = general_conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5


