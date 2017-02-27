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
	def __init__(self,env_name,doubleQ = False, dueling = False,training = True):
		
		#create dirs
		self.trainDir = UtilSettings.trainDir
		self.playDir = UtilSettings.playDir
		self.monitorDir = UtilSettings.monitorDir
		self.experienceDir = UtilSettings.experienceDir
		
		self.folder = env_name + '/'

		self.trainSummaryDir = UtilSettings.trainSummaryDir
		self.playSummaryDir = UtilSettings.playSummaryDir

		pass #TODO -kill	
		self.experienceDir = os.path.join(self.folder,self.experienceDir)
		if not os.path.exists(self.experienceDir):
			os.makedirs(self.experienceDir)
		
		if(not doubleQ and not dueling): #basic DQN
			self.folder +=  'DQN'
				
		elif(doubleQ and not dueling): #doubleDQN
			self.folder += 'doubleDQN'
	    
		elif(not doubleQ and dueling): #duelingDQN
			self.folder += 'duelDQN'
			
		else:#duelDoubleDQN (ddDqn)
			self.folder += 'ddDQN'

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

		if training:
			self.saver_for_train = tf.train.Saver(keep_checkpoint_every_n_hours=2,max_to_keep=2) #will save all the tf graph vars!!!
			self.saver_for_play = tf.train.Saver(tf.trainable_variables(),keep_checkpoint_every_n_hours=2,max_to_keep=2) # used after training 
			
			self.train_writer = tf.summary.FileWriter(self.trainSummaryDir)
		
		#summaries
		self.play_writer = tf.summary.FileWriter(self.playSummaryDir)

		
		
	pass	
	'''saving graph vars'''
	def save_graph(self,sess,step,save2play = False):
		
		print("Saving the training graph @step {}...".format(step))
		checkpoint_file = os.path.join(self.trainDir, 'trainGraph.mz')
		self.saver_for_train.save(sess, checkpoint_file, global_step =step)
		
		if save2play:
			print("Saving the playing graph @step {}...".format(step))
			checkpoint_file = os.path.join(self.playDir, 'playGraph.mz')
			self.saver_for_play.save(sess, checkpoint_file, global_step =step)
	
	def restore_graph(self,sess,forTrain = True):
		if (forTrain):
			latest_checkpoint = tf.train.latest_checkpoint(self.trainDir)
			if latest_checkpoint:
				print("Loading latest training graph checkpoint {}...\n".format(latest_checkpoint))
				#self.saver_for_train.restore(sess, latest_checkpoint)
				t_restore = tf.train.import_meta_graph(latest_checkpoint +'.meta')
				t_restore.restore(sess,latest_checkpoint)
				pass # https://www.tensorflow.org/programmers_guide/meta_graph
				
			else:
				print("No pre-trained model found...Training from scratch :(")
		else:

			latest_checkpoint = tf.train.latest_checkpoint(self.playDir)
			assert latest_checkpoint, "No model was saved for playing :("

			print("Loading latest playing graph checkpoint  {}...\n".format(latest_checkpoint))
			restore = tf.train.import_meta_graph(latest_checkpoint +'.meta')
			restore.restore(sess,latest_checkpoint)

			
	
	def summary_board(self,sess,step,forTrain = True):
		
		'''save summaries so that they can be viewed via tensorboard'''

		summary_merger = tf.summary.merge_all()
		summary_str = sess.run(summary_merger)
		
		print("Updating TensorBoard summaries...@step {}...".format(step))
		
		if(forTrain):
			self.train_writer.add_summary(summary_str, step)
			self.train_writer.flush()
		else:
			self.play_writer.add_summary(summary_str, step)
			self.play_writer.flush()

