'''FOR TRAINING RUN THIS'''

import time
import tensorflow as tf
from tqdm import tqdm
from Q_Learner import DQN
from settings import AgentSetting

#gym env name
ENV_NAME = 'Breakout-v0'

#save model every ? episodes
SAVE_EVERY = 5 
#max update steps to run
MAX_STEPS = AgentSetting.training_steps
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
		episodeNum = dqn.training_episodes #current episode as tensor

		print("Starting training at Episode {} and Step {}...".format(episodeNum.eval(),step.eval()))
		

		with tqdm(initial = step.eval(),total= MAX_STEPS) as curStep:
			
			while (step.eval() < MAX_STEPS):

				cumulatedSteps = step.eval()
				curEpNum = episodeNum.eval() 
				
				dqn.learning(sess) #returns when an episode is done
				
				curStep.update(step.eval())

				print ('Finished Episode no. %d  in %.3f secs...with ::' % (curEpNum, dqn.duration))
				print('Ep.Steps %d...total Ep.Reward = %.2f, total Ep.Loss = %.4f' % (step.eval()-cumulatedSteps, dqn.totalReward, dqn.totalLoss))
				print('Trained for : %d steps and %d episodes in %.3f  hrs' %(step.eval(), curEpNum+1, dqn.training_hrs))			
				
				#saves summmaries for tensorboard
				dqn.util.summary_board(sess,step.eval(),forTrain = TRAINING)
				
				#every some episodes save 
				if ((curEpNum+1) % SAVE_EVERY == 0):

					#saves all tf graph nodes, and if(save2play) saves online weights for evaluation(via mainPlay.py)
					dqn.util.save_graph(sess,step.eval(),save2play = True)

				if (step.eval() >= MAX_STEPS):

					dqn.util.save_graph(sess,step.eval(),save2play = True)
					print('Finished Training for %.0f Frames!...took me %.3f hrs...Now run mainPlay.py and watch me play :)'%(step.eval()* AgentSetting.update_freq,dqn.training_hrs))
					
				
			
#RUN...
main()