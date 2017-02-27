# all env settings GYM
import gym
import tensorflow as tf
import numpy as np
from settings import EnvSetting


class Environment(object):
	
	def __init__(self,env_name, monitorDir):
 		
 		self.env_name = env_name
		env = gym.make(self.env_name)

	 	self.env = gym.wrappers.Monitor(env, monitorDir, video_callable= (lambda count: count % EnvSetting.recEvery == 0),resume = True,force= False,mode = 'evaluation')
		
		self.VALID_ACTIONS = self.numActions()

		self.max_reward = EnvSetting.max_reward
		self.min_reward = EnvSetting.min_reward
		
		self.render_bool = EnvSetting.render 
		self.frame_size = EnvSetting.frame_dim
		
		self.repeat_action = EnvSetting.action_repeat
		
		#3 cores:
		self.observation = None
		self.done = False
		self.reward = 0
		
		with tf.variable_scope('input', initializer = tf.constant_initializer(0)):
			
			self.prev_frame = tf.get_variable(name = 'prev_frame', shape = self.frame_size, dtype = tf.uint8, trainable = False)
			self.cur_frame = tf.get_variable(name = 'cur_frame', shape = self.frame_size, dtype = tf.uint8, trainable = False)


	def numActions(self):
		return self.env.action_space.n

	pass #TODO APPLY
	def takeRandomAction():
		return self.env.action_space.sample()

				
	#take a step given an action
	def step(self, action,sess):
		self.observation, self.reward, self.done, _ = self.env.step(action)
		
		new_prev = tf.assign(self.prev_frame, tf.get_default_graph().get_tensor_by_name('input/cur_frame:0'))
		new_cur = tf.assign(self.cur_frame , self.observation)

		x,y = sess.run([new_cur, new_cur])
		
		#clip all +ve rewards to +1 and all -ve to -1
		self._clip_reward()

		pass
		#self._skipping_steps(action,sess)

		return self.reward, self.done

	# reward clipping
	def _clip_reward(self):

		if self.reward > 0 :
			self.reward =  self.max_reward
		
		elif self.reward < 0 :
			self.reward = self.min_reward
		
		else:
			self.reward = 0.0

	pass #Deprecated
	def _skipping_steps(self,action,sess):
		for _ in range(self.repeat_action -1): # -1 !!!!
			o, r ,d , _ =self.env.step(action) #skipp these frames
			#may arrive at a terminal state.
			if d:
				return self.reset(sess)
				
	#reset env
	def reset(self,sess):
		
		self.observation = self.env.reset()

		new_prev = self.prev_frame.initializer
		new_cur = tf.assign(self.cur_frame , tf.convert_to_tensor(self.observation))

		sess.run([new_prev,new_cur])
		

	#render
	def render(self):

		if self.render_bool:
			self.env.render()

