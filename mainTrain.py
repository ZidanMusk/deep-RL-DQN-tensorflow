'''FOR TRAINING RUN THIS'''

import time
import tensorflow as tf
from tqdm import tqdm
from Q_Learner import DQN


#gym env name
ENV_NAME = 'Breakout-v0'
#max episodes to run
MAX_EPISODES = 1000
#max update steps to run
MAX_STEPS = 250000
#use double DQN
DOUBLE_DQN = False
#use dueling DQN
DUELING_DQN = False

#we are training
TRAINING  = True # training =true , playing= false

def main():

	#init all
	dqn = DQN(ENV_NAME, DOUBLE_DQN, DUELING_DQN, TRAINING)
	
	with tf.Session() as sess:
		
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		
		#tries to restore a previously trained model and resume training
		dqn.util.restore_graph(sess,forTrain = TRAINING)
		
		dqn.fill_memory(sess)
		
		step = tf.train.get_global_step() # gets global step defined in dqn as tensor

		print("Initial global step =  {}".format(step.eval()))
		
		for ep in tqdm(xrange(MAX_EPISODES)):# for episodes
			
			print("Episode no. {} :".format(ep))
			
			dqn.learning(sess) #an episode is done
			
			print('Step %d: totalEpReward = %.2f , totLoss = %.4f  (%.3f sec)' % (step.eval(), dqn.totalReward,dqn.totalLoss,dqn.duration))
			print('Trained for  %.3f  hrs' %(dqn.training_hrs))
			
			#call util summmaries
			dqn.util.summary_board(sess,step.eval(),forTrain = TRAINING)
			
			#if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
			if ep % 2 == 0 or (ep + 1) == MAX_EPISODES:
				dqn.util.save_graph(sess,step.eval(),save2play = True)
				print('Trained for  %.3f  hrs' %(dqn.training_hrs))
			

			
#RUN...
main()