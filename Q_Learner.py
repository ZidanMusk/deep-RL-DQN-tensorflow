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
from settings import AgentSetting, ArchitectureSetting


class DQN(object):

	def __init__(self,env_name, doubleQ = False, dueling = False, perMemory = False, training = True, watch = False ):

		self.agentSteps = tf.Variable(0, trainable=False,name='agentSteps')
		self.agentStepsUpdater = self.agentSteps.assign_add(1)

		# keep in order
		self.util = Utility(env_name, doubleQ, dueling, training)
		self.env = Environment(env_name, self.util.monitorDir)
		self.state_process = StateProcessor()
		self.num_action = self.env.VALID_ACTIONS
		self.deepNet = Brain(self.num_action, training)

		self.net_feed = self.deepNet.nn_input
		self.onlineNet = self.deepNet.Q_nn(forSess=True)
		#self.eee = self.add
		self.actions = np.arange(self.num_action)
		self.no_op_max = AgentSetting.no_op_max
		self.startTime = 0.0
		self.duration = 0.0

		self.totalReward = 0.0
		self.countR = 0
		self.training = training
		self.doubleQ = doubleQ
		self.rendering = watch
		pass
		print ("POSSIBLE ACTIONS :", self.actions)

		if (dueling or perMemory):
			raise Exception('Dueling/PER DQN is under construction.')

		if training:

			self.updates = 0
			self.totalLoss = 0.0
			self.countL = 0

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

			#self.e_decay_rate = (self.e_greedy_init - self.e_greedy_final) / self.e_final_at

			self.epsilon = tf.Variable(0.0, trainable = False, dtype = tf.float32, name = "epsilon")
			self.epsilonHolder = tf.placeholder(dtype = tf.float32)
			self.epsilonUpdater = self.epsilon.assign(self.epsilonHolder)
			
			self.replay_strt_size = AgentSetting.replay_strt_size

			self.global_step =  tf.Variable(0, trainable=False,name='global_step')

			pass #per?
			self.replay_memory = ExperienceMemory(ArchitectureSetting.in_shape, self.replay_memorySize)
			pass

			self.training_hrs = tf.Variable(0.0, trainable=False,name='training_hrs')
			self.training_episodes = tf.Variable(0,trainable = False , name = "training_episodes")

			self.training_hrsHolder = tf.placeholder(dtype = tf.float32)
			self.training_hrsUpdater = self.training_hrs.assign_add((self.training_hrsHolder / 60.0) / 60.0)
			self.training_episodesUpdater = self.training_episodes.assign_add(1)

			self.targetNet = self.deepNet.T_nn(forSess=True)

			if doubleQ:
				'''DoubleQ aims to reduce overestimations of Q-values by decoupling action selection
					from action evaluation in target calculation'''
				# if double
				# 1- action selection using Q-net(online net)
				self.selectedActionIndices = tf.argmax(self.onlineNet, axis=1)
				self.selectedAction = tf.one_hot(indices=self.selectedActionIndices, depth=self.num_action,
												 axis=-1, dtype=tf.float32, on_value=1.0, off_value=0.0)
				# 2- action evaluation using T-net (target net)
				self.nxtState_qValueSelected = tf.reduce_sum(tf.multiply(self.targetNet, self.selectedAction),
															 axis=1)  # element wise
			else:
				# else
				# 1,2- make a one step look ahead and follow a greed policy
				self.nxtState_qValueSelected = tf.reduce_max(self.targetNet, axis=1)

			#3- td-target
			self.td_targetHolder = tf.placeholder(shape=[self.minibatch], name='td-target', dtype=tf.float32)

			#4- current state chosen action value

			self.actionBatchHolder = tf.placeholder(dtype=tf.uint8)
			self.chosenAction = tf.one_hot(indices=self.actionBatchHolder, depth=self.num_action, axis=-1,
										   dtype=tf.float32, on_value=1.0,
										   off_value=0.0)

			self.curState_qValueSelected = tf.reduce_sum(tf.multiply(self.onlineNet, self.chosenAction),
														 axis=1)  # elementwise

			pass #clip
			self.delta = tf.subtract(self.td_targetHolder, self.curState_qValueSelected)
			self.clipped_loss = tf.where(tf.abs(self.delta) < 1.0,
										  0.5 * tf.square(self.delta),
										  tf.abs(self.delta) - 0.5, name='clipped_loss')

			self.loss = tf.reduce_mean(self.clipped_loss, name='loss')

			#$self.loss = tf.reduce_mean(tf.squared_difference(self.td_targetHolder, self.curState_qValueSelected))
			pass
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=self.momentum,
													   epsilon=1e-10)
			self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

			pass  # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
			# self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=self.momentum, epsilon=1e-10)
			# self.train_step = self.optimizer.minimize(self.loss,global_step = self.global_step)

		else:
			self.epsilon = tf.constant(AgentSetting.epsilon_eval,dtype=tf.float32)

		#finallizee
		self.util.summANDsave(self.training)


	#fill memory
	def fill_memory(self,sess,reloadM):

		self.env.reset(sess)
		
		if not reloadM:
			print ('Initializing my experience memory...')
		else:
			print('Restoring my experience memory (naive solution!)...')

		state = self.state_process.get_state(sess)
		done = False
		for v in tqdm(range(self.replay_strt_size)):

			if not reloadM:
				#select  an action randomly
				action = self.env.takeRandomAction()
			else:
				action = self.behaviour_e_policy(state,sess)

			reward , done = self.env.step(action,sess)

			nxt_state = self.state_process.get_state(sess)
			
			experience = (state , action , reward, done , nxt_state)
			self.replay_memory.add(experience)

			if done:
				self.env.reset(sess)
				state = self.state_process.get_state(sess)
			else:
				state = nxt_state
		pass
		print ("Waiting for current episode to be terminated...")

		while not done:
			action = self.env.takeRandomAction()
			reward , done = self.env.step(action,sess)


	def _epsilonDecay(self,sess):

		pass
		eps = self.e_greedy_final + max(0,(self.e_greedy_init - self.e_greedy_final) * (self.e_final_at - self.agentSteps.eval()) / self.e_final_at)

		sess.run(self.epsilonUpdater, feed_dict={self.epsilonHolder: eps})


	#Return the chosen action!
	def behaviour_e_policy(self,state,sess):

		#decay eps and calc prob for actions
		action_probs = (np.ones(self.num_action, dtype =float) * self.epsilon.eval() ) / self.num_action

		q_val = sess.run(self.onlineNet, feed_dict = { self.net_feed : np.expand_dims(state,0)})

		greedy_choice = np.argmax(q_val)
		
		action_probs[greedy_choice] += 1.0 - self.epsilon.eval()

		action = np.random.choice(self.actions, p=action_probs)
		
		pass
		#decay epsilon
		#if self.training:
		#	self._epsilonDecay(sess)

		return action
	
	
	#Playing
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
				self.summaries(sess)
				break #end of episode
			else:
				state = nxt_state
			pass
			if (self.rendering):
				self.env.render()


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

		no_op = 0
		for _ in itertools.count():

			#take action
			action = self.behaviour_e_policy(state,sess)
			#step and observe
			reward , done = self.env.step(action,sess)
			#inc agent steps
			sess.run(self.agentStepsUpdater)
			#decay epsilon after every step
			self._epsilonDecay(sess)

			pass
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

			if( self.agentSteps.eval() % self.update_freq == 0):
				
				#sample  a minibatch
				state_batch, action_batch, reward_batch, done_batch, nxt_state_batch = self.replay_memory.sample(self.minibatch)

				nxtStateFeedDict = {self.net_feed : nxt_state_batch}

				nxtQVal = sess.run(self.nxtState_qValueSelected, feed_dict = nxtStateFeedDict)

				#compute td-target
				td_target = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * nxtQVal

				curStateFeedDict = {self.net_feed: state_batch, self.actionBatchHolder : action_batch, self.td_targetHolder : td_target }
				#run...run...run
				loss, _ = sess.run([self.loss,self.train_step],feed_dict = curStateFeedDict)
				#print ("loss %.5f at step %d" %(loss, self.global_step.eval()))


				#stats
				self.totalLoss += loss
				self.countL +=1
				self.updates +=1 #num of updates made per episode 

			pass #TRY self.global_step.eval()
			if ( self.agentSteps.eval() % self.t_net_update_freq == 1 ):

				sess.run(self.deepNet.updateTparas(True))
				print("Target net parameters updated!")
			pass
			if done:
				
				self.duration = round(time.time() - self.startTime, 3) #secs
				sess.run([self.training_hrsUpdater, self.training_episodesUpdater], feed_dict = { self.training_hrsHolder : self.duration})

				#update tf board every episode
				self.summaries(sess)
				
				break #end of episode
			else:
				state = nxt_state

			pass
			if(self.rendering):
				self.env.render()

	pass #TO DO -> sample of Q-action values summaries

	def summaries(self,sess):
		#print "in summaries!"
		#basics
		listy = {'totReward' : self.totalReward, 'avgReward' : (self.totalReward / self.countR) , 'epDur' : self.duration  }

		if self.training:
			listy.update({"totLoss" : self.totalLoss , "avgLoss" : (self.totalLoss/self.countL), 'epUpdates' : self.updates })

		self.util.summary_board(sess,self.agentSteps.eval(), listy, self.training)
