# timming
# save and load of models
# summaries tf 4 tensorboard

import os
import sys
import time
import tensorflow as tf
from settings import UtilSettings
if "../" not in sys.path:
  sys.path.append("../")

class Utility(object):


	def __init__(self,env_name,doubleQ = False, dueling = False, perMem = False, training = True):

		#create dirs
		self.trainDir = UtilSettings.trainDir
		self.playDir = UtilSettings.playDir
		self.monitorDir = UtilSettings.monitorDir
		#self.experienceDir = UtilSettings.experienceDir

		self.folder = env_name + '/'

		self.trainSummaryDir = UtilSettings.trainSummaryDir
		self.playSummaryDir = UtilSettings.playSummaryDir

		#pass #TODO -kill
		#self.experienceDir = os.path.join(self.folder,self.experienceDir)
		#if not os.path.exists(self.experienceDir):
		#	os.makedirs(self.experienceDir)

		if(perMem):
			self.folder += 'withPrioritizedReplay'
		else:
			self.folder += 'withRandomReplay'

		if(not doubleQ and not dueling): #basic DQN
			self.folder +=  'DQN'

		elif(doubleQ and not dueling): #doubleDQN
			self.folder += 'DoubleDQN'

		elif(not doubleQ and dueling): #duelingDQN
			self.folder += 'DuelDQN'

		else:#duelDoubleDQN (ddDqn)
			self.folder += 'DoDlDQN'

		if training:
			self.monitorDir += '/trainingVids'
		else:
			self.monitorDir += '/playingVids'

		self.monitorDir = os.path.join(self.folder,self.monitorDir)
		self.trainDir = os.path.join(self.folder,self.trainDir)
		self.playDir = os.path.join(self.folder,self.playDir)
		#summaries
		self.trainSummaryDir =  os.path.join(self.folder,self.trainSummaryDir)
		self.playSummaryDir =  os.path.join(self.folder,self.playSummaryDir)

		if not os.path.exists(self.monitorDir):
			os.makedirs(self.monitorDir)
		if not os.path.exists(self.trainDir):
			os.makedirs(self.trainDir)
		if not os.path.exists(self.playDir):
			os.makedirs(self.playDir)

		#summaries
		if not os.path.exists(self.trainSummaryDir):
			os.makedirs(self.trainSummaryDir)
		if not os.path.exists(self.playSummaryDir):
			os.makedirs(self.playSummaryDir)


	def summANDsave(self,training = True):

		with tf.name_scope('saversANDsummaries'):

			if training:
				#saver
				self.saver_for_train = tf.train.Saver(keep_checkpoint_every_n_hours=2,
													  max_to_keep=1)  # will save all the tf graph vars!!!
				self.saver_for_play = tf.train.Saver(tf.trainable_variables(), keep_checkpoint_every_n_hours=2,
													 max_to_keep=10)  # used after training
				self.train_writer = tf.summary.FileWriter(self.trainSummaryDir)

				self.latest_checkpoint = tf.train.latest_checkpoint(self.trainDir)

				#summaries
				# loss
				self.lossTotalSummaryHolder = tf.placeholder(dtype = tf.float16)
				self.lossTotalSummary = tf.summary.scalar('total Loss per episode', self.lossTotalSummaryHolder)
				self.lossAvgSummaryHolder = tf.placeholder(dtype = tf.float16)
				self.lossAvgSummary = tf.summary.scalar('Avg.Loss per episode', self.lossAvgSummaryHolder)
				self.episodeUpdatesHolder = tf.placeholder(dtype = tf.uint16)
				self.episodeUpdates = tf.summary.scalar('Episode updates', self.episodeUpdatesHolder)

			else:

				self.latest_checkpoint = tf.train.latest_checkpoint(self.playDir)

			# reward
			self.rewardTotalSummaryHolder = tf.placeholder(dtype=tf.float16)
			self.rewardTotalSummary = tf.summary.scalar('total Reward per episode', self.rewardTotalSummaryHolder)
			self.rewardAvgSummaryHolder = tf.placeholder(dtype=tf.float16)
			self.rewardAvgSummary = tf.summary.scalar('Avg.Reward per episode', self.rewardAvgSummaryHolder)
			self.episodeDurSummaryHolder = tf.placeholder(dtype=tf.float16)
			self.episodeDurSummary = tf.summary.scalar('Episode duration', self.episodeDurSummaryHolder)

			#savers
			self.play_writer = tf.summary.FileWriter(self.playSummaryDir)
			#merger
			self.summary_merger = tf.summary.merge_all()


	'''saving graph vars'''
	def save_graph(self,sess,step,save2play = False):
		
		print("Saving the training graph @step {}...".format(step))
		checkpoint_file = os.path.join(self.trainDir, 'trainGraph.mz')
		self.saver_for_train.save(sess, checkpoint_file, global_step =step)
		
		if save2play:

			print("Saving the playing graph @step {}...".format(step))
			checkpoint_file = os.path.join(self.playDir, 'playGraph.mz')
			self.saver_for_play.save(sess, checkpoint_file, global_step =step)


	'''restoring graph vars, returns true if restored successfully in training mode'''
	def restore_graph(self,sess,forTrain = True):

		if (forTrain):

			if self.latest_checkpoint:

				print("Loading latest training graph checkpoint {}...\n".format(self.latest_checkpoint))
				t_restore = tf.train.import_meta_graph(self.latest_checkpoint +'.meta')
				t_restore.restore(sess,self.latest_checkpoint)
				pass # https://www.tensorflow.org/programmers_guide/meta_graph
				reloadMem = True
				
			else:
				print("No pre-trained model found...Training from scratch :(")
				reloadMem = False

			return reloadMem
		
		else:

			assert self.latest_checkpoint, "No model was saved for playing :("

			print("Loading latest playing graph checkpoint  {}...\n".format(self.latest_checkpoint))
			restore = tf.train.import_meta_graph(self.latest_checkpoint +'.meta')
			restore.restore(sess,self.latest_checkpoint)


	'''save summaries so that they can be viewed via tensorboard'''
	def summary_board(self,sess,step,sumList,forTrain = True):

		feed_dict = {self.rewardTotalSummaryHolder : sumList['totReward'] , self.rewardAvgSummaryHolder : sumList['avgReward'] , self.episodeDurSummaryHolder : sumList['epDur']}

		if forTrain:

			feed_dict.update({self.lossTotalSummaryHolder : sumList['totLoss'], self.lossAvgSummaryHolder : sumList['avgLoss'], self.episodeUpdatesHolder : sumList['epUpdates']})
			summary_str = sess.run(self.summary_merger, feed_dict=feed_dict)
			self.train_writer.add_summary(summary_str, step)
			self.train_writer.flush()

		else:

			summary_str = sess.run(self.summary_merger, feed_dict=feed_dict)
			self.play_writer.add_summary(summary_str, step)
			self.play_writer.flush()

		print("Updating TensorBoard summaries...@step {}...".format(step))
