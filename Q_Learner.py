#deep Q-learning 
import time
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import Utility
from AgentBrain import Brain
from environment import Environment
from memory import ExperienceMemory
from StateProcessor import StateProcessor
from settings import AgentSetting, EnvSetting

#TODO clip loss b/n [-1,1]

class DQN( Brain, StateProcessor, Environment, ExperienceMemory):

	def __init__(self,env_name,doubleQ = False, dueling = False, training = True):

		
		if training:

			self.minibatch = AgentSetting.minibatch
			self.replay_memorySize = AgentSetting.replay_memory
			self.t_net_update_freq = AgentSetting.t_net_update_freq
			self.discount_factor = AgentSetting.discount_factor
			self.update_freq = AgentSetting.update_freq
			
			self.learning_rate = AgentSetting.learning_rate
			self.momentum = AgentSetting.momentum

			self.e_greedy_init = AgentSetting.e_greedy_init
			self.e_greedy_final = AgentSetting.e_greedy_final
			self.e_final_at = AgentSetting.e_final_at

			self.e_decay_rate = tf.Variable((self.e_greedy_init - self.e_greedy_final) / self.e_final_at, dtype= tf.float32, name = "epsilonDecay",trainable = False)
			self.epsilon = tf.Variable(self.e_greedy_init, dtype = tf.float32, name = "epsilon")
			
			self.replay_strt_size = AgentSetting.replay_strt_size
			self.no_op_max = AgentSetting.no_op_max


			self.global_step =  tf.Variable(0, trainable=False,name='global_step')
			self.replay_memory = ExperienceMemory(self.replay_memorySize)
			
			self.td_target = tf.placeholder(shape =[self.minibatch] ,name = 'td-target', dtype = tf.float32)
			self.selectedActionValue = tf.get_variable(shape = [self.minibatch], name = 'selectedQnetVals', dtype = tf.float32, trainable = True)
			self.loss = tf.reduce_mean(tf.squared_difference(self.td_target, self.selectedActionValue))
		
			self.totalLoss = 0.0
			self.countL = 0

			pass #https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=self.momentum, epsilon=1e-10)
			self.train_step = self.optimizer.minimize(self.loss,global_step = self.global_step)
			
			self.training_hrs = tf.Variable(0.0, trainable=False,name='training_hrs')
			self.training_episodes = tf.Variable(0,trainable = False , name = "training_episodes")

		else:
			self.epsilon = tf.constant(AgentSetting.epsilon_eval,dtype=tf.float32)
		
		#keep in order
		self.util = Utility(env_name, doubleQ, dueling,training)
		self.env = Environment(env_name,self.util.monitorDir)

		self.num_action = self.env.VALID_ACTIONS
		self.actions = np.arange(self.num_action)
		
		pass
		print ("POSSIBLE ACTIONS :", self.actions)
		
		self.deepNet = Brain(self.num_action,training)
		
		self.state_process = StateProcessor()

		self.net_feed = self.deepNet.nn_input
		
		
		self.training = training

		self.totalReward = 0.0
		self.countR = 0		
		
		#timing
			
		self.startTime = 0.0 
		self.duration = 0.0
		self.updates = 0
		
		if dueling:
			raise Exception('Dueling DQN is under construction.') 
		self.doubleQ = doubleQ

	pass 
	#fill memory
	def fill_memory(self,sess):

		self.env.reset(sess)
		print ('Initializing my experience memory...')
		
		state = self.state_process.get_state(sess)
		done = False
		action = 0
		for v in tqdm(range(self.replay_strt_size)):
 
			#select  an action randomly
			action = self.env.takeRandomAction()
			#print action
			reward , done = self.env.step(action,sess)

			
			nxt_state = self.state_process.get_state(sess)
			
			experience = (state , action , reward, done , nxt_state)
			self.replay_memory.add(experience)
			#self.env.render()
			if done:
				self.env.reset(sess)
				state = self.state_process.get_state(sess)
			else:
				state = nxt_state
		pass
		print ("Waiting for current episode to be terminated...")
		while not done:
			#select  an action randomly
			action = self.env.takeRandomAction()
			reward , done = self.env.step(action,sess)


	def _epsilonDecay(self,sess):

		if(self.epsilon.eval() > self.e_greedy_final):
			eps = self.epsilon.eval() - self.e_decay_rate.eval()
			updateEpsilon = self.epsilon.assign(eps)
			sess.run(updateEpsilon)
			
		else:	
			print ("epsilon reached its final value @ {} ".format(self.epsilon.eval()))
	
	#Return the chosen action!
	def behaviour_e_policy(self,state,sess):
		#decay eps and calc prob for actions
		action_probs = (np.ones(self.num_action, dtype =float) * self.epsilon.eval() ) / self.num_action
		pass
		q_val = self.deepNet.Q_nn(forSess= True).eval(feed_dict = { self.net_feed : np.expand_dims(state,0)})
		
		greedy_choice = np.argmax(q_val)
		
		action_probs[greedy_choice] += 1.0 - self.epsilon.eval()

		action = np.random.choice(self.actions, p=action_probs)
		
		#decay epsilon
		if self.training:
			self._epsilonDecay(sess)

	
		return action
	
	
	pass #Playing
	def playing(self,sess):
		self.totalReward = 0.0
		self.countR = 0

		self.startTime = time.time()
		self.env.reset(sess)

		state = self.state_process.get_state(sess)

		for t in itertools.count():

			action = self.behaviour_e_policy(state,sess)
			reward , done = self.env.step(action,sess)
			self.totalReward += reward
			self.countR += 1
			nxt_state = self.state_process.get_state(sess)

			print("playing well as much as you trained me :)")
			
			if done:
				
				self.duration = round(time.time() - self.startTime, 3)
				self.summaries()
				break #end of episode
			else:
				state = nxt_state

	pass
	def learning(self,sess):
		#loop for one episode
		#reset vars
		self.totalLoss =0.0
		self.countL = 0
		self.totalReward = 0.0
		self.countR = 0
		self.updates = 0

		self.startTime = time.time()
		self.env.reset(sess)
	
		state = self.state_process.get_state(sess)
		#print state
		no_op = 0
		for t in itertools.count():
			
			action = self.behaviour_e_policy(state,sess)
			reward , done = self.env.step(action,sess)
			
			if(action == 0):
				no_op += 1
			
			pass #can't force episode to end
			#if(no_op == self.no_op_max): #end this boring episode
			#	done = True
		
			self.totalReward += reward
			self.countR += 1

			nxt_state = self.state_process.get_state(sess) 
			
			experience = (state , action , reward, done , nxt_state)
			self.replay_memory.add(experience)
		
			if((t+1) % self.update_freq == 0):
				
				
				#sample  a minibatch
				state_batch, action_batch, reward_batch, done_batch, nxt_state_batch = self.replay_memory.sample(self.minibatch)
				pass

				if self.doubleQ:
					'''DoubleQ aims to reduce overestimations of Q-values by decoupling action selection from action evaluation in target calculation''' 
					
					#1- action selection using Q-net(online net)
					nxtState_qValue4Selection = self.deepNet.Q_nn(forSess = True).eval(feed_dict = { self.net_feed : nxt_state_batch })#online network	
					selectedAction = tf.one_hot( indices = tf.argmax(nxtState_qValue4Selection, axis = 1), depth = self.num_action, axis =-1,dtype = tf.float32,on_value = 1.0 , off_value = 0.0).eval()

					#2- action evaluation using T-net (target net)
					nxtState_qValue4Evaluation = self.deepNet.T_nn(forSess = True).eval(feed_dict = { self.net_feed : nxt_state_batch }) #target network
					nxtState_qValueSelected =tf.reduce_sum(tf.multiply( nxtState_qValue4Evaluation, selectedAction),axis =1) #elementwise 
				
				else:
					#1- make a one step look ahead
					nxtState_qValue = self.deepNet.T_nn(forSess = True).eval(feed_dict = { self.net_feed : nxt_state_batch }) #target network

					pass 
					#2- be greedy and select nxt action (of nxt state)
					nxtState_qValueSelected = tf.reduce_max(nxtState_qValue, axis= 1)


				#3 - compute td-target
				td_target = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * nxtState_qValueSelected.eval()
				
				

				#4 - current state Q-values predictions #trainable!
				curState_qValue = self.deepNet.Q_nn(forSess = True)
				
				#5 - current state chosen action values
				chosenAction = tf.one_hot( indices = action_batch , depth = self.num_action, axis =-1,dtype = tf.float32,on_value = 1.0 , off_value = 0.0) 
				
				curState_qValueSelected = tf.reduce_sum(tf.multiply(curState_qValue, chosenAction),axis =1) #elementwise 

				updateSelectedActionValue = self.selectedActionValue.assign(curState_qValueSelected) 

				pass
				#6 - run...run...run
				ss, loss, _ = sess.run([updateSelectedActionValue,self.loss,self.train_step],feed_dict = {self.net_feed : state_batch , self.td_target : td_target})
				print ("loss %.3f at step %d" %(loss, self.global_step.eval()))
				
				#stats
				self.totalLoss += loss
				self.countL +=1
				self.updates +=1 #num of updates made per episode 
				
				pass #update T to Q
				if (self.global_step.eval() % self.t_net_update_freq == 0 ):
					self.deepNet.updateTparas(sess)

			if done:
				
				self.duration = round(time.time() - self.startTime, 3) #secs
				self.training_hrs = tf.add(self.training_hrs,((self.duration/60.0)/60.0)).eval()
				incEpisode = self.training_episodes.assign_add(1)
				incEpisode.eval()

				#update tf board
				self.summaries()
				
				break #end of episode
			else:
				state = nxt_state

			#self.env.render()

			
	pass #TO DO -> sample of Q-action values summaries

	def summaries(self):
		#print "in summaries!"
		with tf.name_scope('summaries'):
			#loss
			if self.training and self.countL > 0:
				tf.summary.scalar('total Loss per episode', self.totalLoss).eval()
				tf.summary.scalar('Avg.Loss per episode', self.totalLoss/self.countL).eval()
				
				tf.summary.scalar('Episode updates', self.updates).eval()
			#reward
			if self.countR > 0:
				tf.summary.scalar('total Reward per episode', self.totalReward).eval()
				tf.summary.scalar('Avg.Reward per episode', self.totalReward/self.countR).eval()
			
			tf.summary.scalar('Episode duration', self.duration).eval()



	


	
