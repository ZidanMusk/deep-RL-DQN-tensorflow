'''FOR TRAINING RUN THIS'''

import time
import tensorflow as tf
from tqdm import tqdm
from Q_Learner import DQN


#gym env name
ENV_NAME = 'Breakout-v0'
#max episodes to run
MAX_EPISODES = 10

#use double DQN
DOUBLE_DQN = False
#use dueling DQN
DUELING_DQN = False

#we are PLAYING
TRAINING  = False # training =true , playing= false

def main():

	dqn = DQN(ENV_NAME, DOUBLE_DQN, DUELING_DQN, TRAINING)
	
	with tf.Session() as sess:
		
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		#tries to restore a trained model and play!
		dqn.util.restore_graph(sess,forTrain = TRAINING)

		for ep in tqdm(xrange(MAX_EPISODES)):# for episodes
			
			print("Episode no. {} :".format(ep))
			
			dqn.playing(sess)
			print('Episode %d: totalEpReward = %.2f , took: %.3f mins' % (ep, dqn.totalReward,dqn.duration/60.0))

			dqn.util.summary_board(sess,ep,forTrain = TRAINING)



#RUN...
main()