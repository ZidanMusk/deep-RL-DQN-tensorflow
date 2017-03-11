'''FOR PLAYING/EVALUATION RUN THIS'''

import time
from tqdm import tqdm
import tensorflow as tf
from Q_Learner import DQN


#gym env name
ENV_NAME = 'Breakout-v0'
#max episodes to run
MAX_EPISODES = 10

#use double DQN
DOUBLE_DQN = True
#use dueling DQN
DUELING_DQN = True
#prioritized experience replay PER
PER = True

#we are PLAYING
TRAINING  = False # training =true , playing= false

#WATCH Playing LIVE
RENDER = True

def main():

	dqn = DQN(ENV_NAME, DOUBLE_DQN, DUELING_DQN, PER, TRAINING, RENDER)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	with tf.Session() as sess:

		sess.run(init_op)
		#tries to restore a trained model and play!
		dqn.util.restore_graph(sess,forTrain = TRAINING)

		for ep in tqdm(range(MAX_EPISODES)):# for episodes
			
			print("Episode no. {} :".format(ep))
			
			dqn.playing(sess)

			print('Episode %d: totalEpReward = %.2f , took: %.3f mins' % (ep, dqn.totalReward,dqn.duration/60.0))


#RUN...
if __name__ == "__main__":
	main()

main()