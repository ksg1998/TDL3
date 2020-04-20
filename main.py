# https://github.com/ckmarkoh/GAN-tensorflow
import tensorflow as tf
import numpy as np
import scipy 
import imageio
import os
import shutil
from PIL import Image
import time
import random
import sys


from layers import *
from model import *

batch_size = 1
pool_size = 50

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"


temp_check = 0



max_epoch = 1
max_images = 100

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64

class cyclepix():

	# preparing the input pipeline
	def input(self):

		faceFiles = tf.train.match_filenames_once(
			"./input/faces/train_photos/*.jpg")
		self.qlengthFace = tf.size(faceFiles)
		sketchFiles = tf.train.match_filenames_once(
			"./input/faces/train_sketches/*.jpg")
		self.qlengthSketch = tf.size(sketchFiles)

		faces_queue = tf.train.string_input_producer(faceFiles)

		sketches_queue = tf.train.string_input_producer(sketchFiles)

		allImages = tf.WholeFileReader()
		# storing temp files,
		_, imageFileFaces = allImages.read(faces_queue)
		_, imageFileSketches = allImages.read(sketches_queue)

		# try different resize options here
		self.faceImage = tf.subtract(tf.div(tf.image.resize_images(
			tf.image.decode_jpeg(imageFileFaces), [256, 256]), 127.5), 1)

		self.sketchImage = tf.subtract(tf.div(tf.image.resize_images(
			tf.image.decode_jpeg(imageFileSketches), [256, 256]), 127.5), 1)

		# now we have every image as a tensor.

		# Input pipeline complete

	def reading(self, sess):
		# to read in the prepared data pipeline

		# for multithreading for reading from the queue
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		# a = sess.run(self.qlengthFace)
		# b = sess.run(self.qlengthSketch)

		sess.run(self.qlengthFace)
		sess.run(self.qlengthSketch)

		self.generatedFaces = np.zeros(
			(pool_size, 1, img_height, img_width, img_layer))
		self.generatedSketches = np.zeros((
			pool_size, 1, img_height, img_width, img_layer))

		self.faceInput = np.zeros(
			(max_images, batch_size, img_height, img_width, img_layer))
		self.sketchInput = np.zeros(
			(max_images, batch_size, img_height, img_width, img_layer))

		for i in range(max_images):
			image_tensor = sess.run(self.faceImage)
			if (image_tensor.size == img_size * batch_size * img_layer):
				self.faceInput[i] = image_tensor.reshape(
					(batch_size, img_height, img_width, img_layer))

		for i in range(max_images):
			image_tensor = sess.run(self.sketchImage)
			if (image_tensor.size == img_size * batch_size * img_layer):
				self.sketchInput[i] = image_tensor.reshape(
					(batch_size, img_height, img_width, img_layer))

		coord.request_stop()
		coord.join(threads)

	# setting up the model
	def model_setup(self):

		# placeholder(dtype, shape=None, name=None)
		self.inputFace = tf.placeholder(
			tf.float32, [batch_size, img_width, img_height, img_layer], name="inputFace")
		# placeholder(dtype, shape=None, name=None)
		self.inputSketch = tf.placeholder(
			tf.float32, [batch_size, img_width, img_height, img_layer], name="inputSketch")

		# Pool is used to escape computing the loss calculation for each and every image. You calculates loss for any random pic picked from the array

		# None as shape -> 1 D array. Upon giving args it allocates releavant memory to the array
		self.generatedFacesPool = tf.placeholder(
			tf.float32, [None, img_width, img_height, img_layer], name="generatedFacesPool")
		self.generatedSketchesPool = tf.placeholder(
			tf.float32, [None, img_width, img_height, img_layer], name="generatedSketchesPool")

		# NEED TO CONFIRM
		self.global_step = tf.Variable(
			0, name="global_step", trainable=False)

		self.numGenerated = 0

		# FOR ADAPTIVE LEARNING RATE
		self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

		# self.lr = 0.01

		with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:
			self.fakeSketches = resnet_generator_9(
				self.inputFace, name="faceGenerator")
			self.fakeFaces = resnet_generator_9(
				self.inputSketch, name="sketchGenerator")
			self.recFace = discriminator_builder(
				self.inputFace, "faceDisciminator")
			self.recSketch = discriminator_builder(
				self.inputSketch, "sketchDiscriminator")

			# scope.reuse_variables()

			self.fake_rec_Face = discriminator_builder(
				self.fakeFaces, "faceDisciminator")
			self.fake_rec_Sketch = discriminator_builder(
				self.fakeSketches, "sketchDiscriminator")
			self.cyc_Face = resnet_generator_9(
				self.fakeSketches, "sketchGenerator")
			self.cyc_Sketch = resnet_generator_9(
				self.fakeFaces, "faceGenerator")

			# scope.reuse_variables()

			self.fake_pool_rec_Face = discriminator_builder(
				self.generatedFacesPool, "faceDisciminator")
			self.fake_pool_rec_Sketch = discriminator_builder(
				self.generatedSketchesPool, "sketchDiscriminator")

	def lossFunction(self):
		# using MSE as a loss metric

		cyc_loss = tf.reduce_mean(tf.abs(
			self.inputFace-self.cyc_Face)) + tf.reduce_mean(tf.abs(self.inputSketch-self.cyc_Sketch))
		disc_loss_Face = tf.reduce_mean(
			tf.squared_difference(self.fake_rec_Face, 1))
		disc_loss_Sketch = tf.reduce_mean(
			tf.squared_difference(self.fake_rec_Sketch, 1))

		faceGeneratorLoss = cyc_loss*10 + disc_loss_Sketch
		sketchGeneratorLoss = cyc_loss*10 + disc_loss_Face

		faceDiscriminatorLoss = (tf.reduce_mean(tf.square(self.fake_pool_rec_Face)) +
									tf.reduce_mean(tf.squared_difference(self.recFace, 1)))/2.0
		sketchDiscriminatorLoss = (tf.reduce_mean(tf.square(self.fake_pool_rec_Sketch)) +
									tf.reduce_mean(tf.squared_difference(self.recSketch, 1)))/2.0

		optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

		self.model_vars = tf.trainable_variables()

		faceDiscriminator_vars = [
			var for var in self.model_vars if 'faceDisciminator' in var.name]
		faceGenerator_vars = [
			var for var in self.model_vars if 'faceGenerator' in var.name]
		sketchDiscriminator_vars = [
			var for var in self.model_vars if 'sketchDiscriminator' in var.name]
		sketchGenerator_vars = [
			var for var in self.model_vars if 'sketchGenerator' in var.name]

		self.faceDisciminator_trainer = optimizer.minimize(
			faceDiscriminatorLoss, var_list=faceDiscriminator_vars)
		self.sketchDiscriminator_trainer = optimizer.minimize(
			sketchDiscriminatorLoss, var_list=sketchDiscriminator_vars)
		self.faceGenerator_trainer = optimizer.minimize(
			faceGeneratorLoss, var_list=faceGenerator_vars)
		self.sketchGenerator_trainer = optimizer.minimize(
			sketchGeneratorLoss, var_list=sketchGenerator_vars)

		#Summary variables for tensorboard

		self.faceGeneratorLoss_summ = tf.summary.scalar("faceGeneratorLoss", faceGeneratorLoss)
		self.sketchGeneratorLoss_summ = tf.summary.scalar("sketchGeneratorLoss", sketchGeneratorLoss)
		self.faceDiscriminatorLoss_summ = tf.summary.scalar("faceDiscriminatorLoss", faceDiscriminatorLoss)
		self.sketchDiscriminatorLoss_summ = tf.summary.scalar("sketchDiscriminatorLoss", sketchDiscriminatorLoss)

	def save_images(self, sess, epoch):
		if not os.path.exists("./output/imgs"):
			os.makedirs("./output/imgs")

		for i in range(0, 10):
			fakeFaces_temp, fakeSketches_temp, cyc_Faces_temp, cyc_Sketch_temp = sess.run([self.fakeFaces, self.fakeSketches, self.cyc_Face, self.cyc_Sketch], feed_dict={
				self.inputFace: self.faceInput[i], self.inputSketch: self.sketchInput[i]})
			imageio.imwrite("./output/imgs/generatedSketches_" + str(epoch) + "_" + str(i) +
					".jpg", ((fakeFaces_temp[0]+1)*127.5).astype(np.uint8))
			imageio.imwrite("./output/imgs/generatedFaces" + str(epoch) + "_" + str(i) +
					".jpg", ((fakeSketches_temp[0]+1)*127.5).astype(np.uint8))
			imageio.imwrite("./output/imgs/cycFaces_" + str(epoch) + "_" + str(i) +
					".jpg", ((cyc_Faces_temp[0]+1)*127.5).astype(np.uint8))
			imageio.imwrite("./output/imgs/cycSketches_" + str(epoch) + "_" + str(i) +
					".jpg", ((cyc_Sketch_temp[0]+1)*127.5).astype(np.uint8))
			imageio.imwrite("./output/imgs/inputFaces_" + str(epoch) + "_" + str(i) +
					".jpg", ((self.faceInput[i][0]+1)*127.5).astype(np.uint8))
			imageio.imwrite("./output/imgs/inputSketches_" + str(epoch) + "_" + str(i) +
					".jpg", ((self.sketchInput[i][0] + 1) * 127.5).astype(np.uint8))

	# image pooling
	def generatedPool(self, numGenerated, generated, genPool):
		# adding to the image pool
		if(numGenerated < pool_size):
			genPool[numGenerated] = generated
			return generated
		# random replacement of images
		else:
			p = random.random()
			if p > 0.5:
				random_id = random.randint(0, pool_size-1)
				temp = genPool[random_id]
				# pushing in the new image
				genPool[random_id] = generated
				return temp
			else:
				return generated

	''' training function '''

	def train(self):
		self.input()
		self.model_setup()
		self.lossFunction()
		# initialising global variables
		init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])
		# saving and restoring variables for checkpoints
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			self.reading(sess)
			if to_restore:
				chkpt_fname = tf.train.latest_checkpoint(check_dir)
				saver.restore(sess, chkpt_fname)

			writer = tf.summary.FileWriter("./output/2")

			if not os.path.exists(check_dir):
				os.makedirs(check_dir)

			# setting epochs to 100 (reduce if needed)
			for epoch in range(sess.run(self.global_step),100):                
				print ("In the epoch ", epoch)
				saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

				# # Dealing with the learning rate as per the epoch number
				# if(epoch < 100) :
				curr_lr = 0.0002
				# else:
				#     curr_lr = 0.0002 - 0.0002*(epoch-100)/100
				# curr_lr = self.lr

				if(save_training_images):
					self.save_images(sess, epoch)

				# sys.exit()
				for ptr in range(0,max_images):
					print("In the iteration ",ptr)
					print("Starting",time.time()*1000.0)

					# Optimizing the G_A network

					_, fakeSketches_temp, summary_str = sess.run([self.faceGenerator_trainer, self.fakeSketches, self.faceGeneratorLoss_summ],feed_dict={self.inputFace:self.faceInput[ptr], self.inputSketch:self.sketchInput[ptr], self.lr:curr_lr})
					
					writer.add_summary(summary_str, epoch*max_images + ptr)                    
					fakeSketches_temp1 = self.generatedPool(self.numGenerated, fakeSketches_temp, self.generatedSketches)
					
					# Optimizing the D_B network
					_, summary_str = sess.run([self.sketchDiscriminator_trainer, self.sketchDiscriminatorLoss_summ],feed_dict={self.inputFace:self.faceInput[ptr], self.inputSketch:self.sketchInput[ptr], self.lr:curr_lr, self.generatedSketchesPool:fakeSketches_temp1})
					writer.add_summary(summary_str, epoch*max_images + ptr)
					
					
					# Optimizing the G_B network
					_, fakeFaces_temp, summary_str = sess.run([self.sketchGenerator_trainer, self.fakeFaces, self.sketchGeneratorLoss_summ],feed_dict={self.inputFace:self.faceInput[ptr], self.inputSketch:self.sketchInput[ptr], self.lr:curr_lr})

					writer.add_summary(summary_str, epoch*max_images + ptr)
					
					
					fakeFaces_temp1 = self.generatedPool(self.numGenerated, fakeFaces_temp, self.generatedFaces)

					# Optimizing the D_A network
					_, summary_str = sess.run([self.faceDisciminator_trainer, self.faceDiscriminatorLoss_summ],feed_dict={self.inputFace:self.faceInput[ptr], self.inputSketch:self.sketchInput[ptr], self.lr:curr_lr, self.generatedFacesPool:fakeFaces_temp1})

					writer.add_summary(summary_str, epoch*max_images + ptr)
					
					self.numGenerated += 1
					
				sess.run(tf.assign(self.global_step, epoch + 1))
			writer.add_graph(sess.graph)
	
	def test(self):
		print("Testing the model made: ")
		self.input()
		self.model_setup()
		saver = tf.train.Saver()
		init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

		with tf.Session() as sess:
			sess.run(init)
			self.input()
			chkpt_fname = tf.train.latest_checkpoint(check_dir)
			saver.restore(sess, chkpt_fname)
			if not os.path.exists("./output/imgs/test/"):
				os.makedirs("./output/imgs/test/")            
			for i in range(0,100):
				fakeFaces_temp, fakeSketches_temp = sess.run([self.generatedFaces, self.generatedSketches],feed_dict={self.inputFace:self.faceInput[i], self.inputSketch:self.sketchInput[i]})
				imageio.imwrite("./output/imgs/test/generatedSketches_"+str(i)+".jpg",((fakeFaces_temp[0]+1)*127.5).astype(np.uint8))
				imageio.imwrite("./output/imgs/test/generatedFaces_"+str(i)+".jpg",((fakeSketches_temp[0]+1)*127.5).astype(np.uint8))
				imageio.imwrite("./output/imgs/test/inputFaces_"+str(i)+".jpg",((self.faceInput[i][0]+1)*127.5).astype(np.uint8))
				imageio.imwrite("./output/imgs/test/inputSketches"+str(i)+".jpg",((self.sketchInput[i][0]+1)*127.5).astype(np.uint8))
	
def main():
	model = cyclepix()
	if to_train:
		model.train()
	elif to_test:
		model.test()

if __name__ == '__main__':
	main()
