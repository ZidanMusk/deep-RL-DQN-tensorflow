import tensorflow as tf
import numpy as np
from settings import StateProcessorSetting

'''where converting raw frames to states happens''' 

class StateProcessor(object):
	
	def __init__(self):

		self.history = StateProcessorSetting.history_length
		self.dims = StateProcessorSetting.observation_dims
		pass

		#get current,prev frame, set by env
		with tf.variable_scope('input', reuse =True):
			self.cur_frame = tf.get_variable('cur_frame',dtype = tf.uint8)
			self.prev_frame = tf.get_variable('prev_frame',dtype = tf.uint8)

		with tf.variable_scope('input'):
			maxOf2 = tf.maximum(tf.to_float(self.cur_frame), tf.to_float(self.prev_frame))
			toGray = tf.expand_dims(tf.image.rgb_to_grayscale(maxOf2), 0)
			resize = tf.image.resize_bilinear(toGray, self.dims, align_corners=None, name='observation')
			self.observe = tf.div(tf.squeeze(resize), 255.0) 

			self.state = tf.get_variable(name = 'state', shape = [self.dims[0],self.dims[1],self.history], dtype = tf.float32,initializer = tf.constant_initializer(0.0),trainable = False)

	pass
	#ENV WILL ASSIGN prev and cur FRAmes accordingly

	def _process(self,sess):

		processed = sess.run(self.observe)
		
		return processed # 84 x 84
			
		
	def get_state(self, sess):
		#stackkkkk
		self.to_stack = self._process(sess) #84x84

		with tf.variable_scope('input', reuse = True):

			oldState = tf.get_variable(name = 'state', dtype = tf.float32)
			

		#expand dims of to_stack 2 -> 84x84x1
		self.to_stack = tf.expand_dims(self.to_stack,2)
		
		#split along depth!
		self.f3, self.f2, self.f1, _ = tf.split(2,  self.history, oldState) #each is 84x84x1
		
		with tf.variable_scope('input'):

			updateState = self.state.assign(tf.concat(2,[self.to_stack, self.f3, self.f2, self.f1], name='concat'))

		
		return sess.run(updateState)
		