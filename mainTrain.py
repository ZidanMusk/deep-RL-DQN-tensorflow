'''FOR TRAINING RUN THIS'''

import time
import tensorflow as tf
from tqdm import tqdm
from Q_Learner import DQN
from settings import AgentSetting

#gym env name
ENV_NAME = 'Breakout-v0'
#max episodes to run 
#TODO Deprecate it
MAX_EPISODES = 10000
#save model every ? episodes
SAVE_EVERY = 5 
#max update steps to run
MAX_STEPS = AgentSetting.trained_frames / AgentSetting.update_freq
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
			
			print("Episode no. {} :".format(ep+1))
			
			dqn.learning(sess) #an episode is done
			
			print('Step %d: totalEpReward = %.2f , totLoss = %.4f  (%.3f sec)' % (step.eval(), dqn.totalReward,dqn.totalLoss,dqn.duration))
			print('Trained for  %.3f  hrs' %(dqn.training_hrs))
			
			#call util summmaries
			dqn.util.summary_board(sess,step.eval(),forTrain = TRAINING)
			
			if ((ep % SAVE_EVERY == 0) or ((ep + 1) == MAX_EPISODES)):
				dqn.util.save_graph(sess,step.eval(),save2play = True)
				print('Trained for  %.3f  hrs' %(dqn.training_hrs))

			if (step.eval() >= MAX_STEPS):
				dqn.util.save_graph(sess,step.eval(),save2play = True)
				print('Finished Training for %.0f Frames!...took me %.3f hrs...Now run mainPlay.py and watch me play :)'%(step.eval()* AgentSetting.update_freq,dqn.training_hrs))
				break
			

			
#RUN...
main()