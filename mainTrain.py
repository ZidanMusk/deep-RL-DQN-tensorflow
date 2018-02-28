'''FOR TRAINING RUN THIS'''

import time
import tensorflow as tf
from tqdm import tqdm
from Q_Learner import DQN
from settings import AgentSetting

#gym env name
ENV_NAME = 'Pong-v0'

#save model every ? episodes
SAVE_EVERY = 200
#max update steps to run
MAX_STEPS = AgentSetting.training_steps
#use double DQN
DOUBLE_DQN = True
#use dueling DQN
DUELING_DQN = False
#prioritized experience replay PER
PER = False

#we are training
TRAINING  = True # training =true , playing= false

#WATCH Training LIVE
RENDER = False

def main():

	dqn = DQN(ENV_NAME, DOUBLE_DQN, DUELING_DQN,PER, TRAINING, RENDER)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	episodeNum = dqn.training_episodes  # current episode as tensor
	step = dqn.agentSteps
	updates = dqn.global_step
	timer = dqn.training_hrs

	with tf.Session() as sess:

		sess.run(init_op)

		pass
		#tries to restore a previously trained model and resume training
		reloadMemory = dqn.util.restore_graph(sess,forTrain = TRAINING)
		
		dqn.fill_memory(sess, reloadMemory)

		print("Starting training at Episode {} and Step {}...".format(episodeNum.eval(),step.eval()))

		with tqdm(initial = step.eval(),total= MAX_STEPS) as curStep:
			
			while (step.eval() < MAX_STEPS):

				cumulatedSteps = step.eval()
				curEpNum = episodeNum.eval()

				dqn.learning(sess) #returns when an episode is done

				print ('Finished Episode no. {} with ep_greedy = {}...'.format (curEpNum, dqn.epsilon.eval()))
				print('Ep.AgentSteps = %d...Ep.Updates = %d...total Ep.Reward = %.2f' % (step.eval()-cumulatedSteps, dqn.updates, dqn.totalReward))
				print('Learned with %d updates...%d episodes...in %.3f  hrs' %(updates.eval(), curEpNum+1, timer.eval()))

				curStep.update(step.eval() - cumulatedSteps)

				#every some episodes save 
				if ((curEpNum+1) % SAVE_EVERY == 0):

					#saves all tf graph nodes, and if(save2play) saves online weights for evaluation(via mainPlay.py)
					dqn.util.save_graph(sess,step.eval(),save2play = True)

				if (step.eval() >= MAX_STEPS):

					dqn.util.save_graph(sess,step.eval(),save2play = True)
					print('Finished Training by performing %d Updates and %d Steps!...took me %.3f hrs...Now run mainPlay.py and watch me play :)'%(dqn.global_step.eval(),step.eval(), timer.eval()))

#RUN...
if __name__ == "__main__":
	main()
