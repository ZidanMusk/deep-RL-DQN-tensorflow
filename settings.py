
'''All parameters are set here'''

class AgentSetting():
	
	minibatch = 16 #32
	
	''''commented values are the used ones in the nature paper, use them if you got that much of computing power'''
	replay_strt_size = 32 #50000 # at start use randm policy for this time to fill memory b4 learning
	replay_memory = 128 #1000000
	
	t_net_update_freq = 100 #10000
	
	discount_factor = 0.99
	
	update_freq = 4 #update net every 4 actions
	
	
	learning_rate = 0.00025
	#RMSProp #deprecated
	#grad_momentum = 0.95
	#sq_grad_momentum = 0.95
	#min_sq_grad = 0.01

	#epsilon
	e_greedy_init = 1.0
	e_greedy_final = 0.1

	e_final_at = 1000000 #frames

	no_op_max = 30 #max of no-op action

class ArchitectureSetting():
	
	#layer 1
	in_shape = [84,84,4]
	f1_no  = 32
	f1_size = [8,8]
	stride1 = 4
	#layer2
	f2_no = 64
	f2_size = [4,4]
	stride2 = 2
	#layer3
	f3_no = 64
	f3_size = [3,3]
	stride3 = 1
	#layer4 fc
	nodes = 512



class StateProcessorSetting():
	
	history_length = 4
	observation_dims = [84, 84]
		
class EnvSetting():
	
	recEvery = 2 #rec every ? step
	'''gym:Each action is repeatedly performed for a duration of k frames, where k is uniformly sampled from {2, 3, 4}.'''
	action_repeat = 1 #4 
	
	#4 reward clipping
	max_reward =  1.0
	min_reward = -1.0

	#display
	frame_dim = [210,160,3]
	render = True


class UtilSettings():
	'''useful paths '''

	trainDir = 'graphVars/train' #dir for checkpoints during training
	playDir = 'graphVars/play' #dir for model weights for playing/eval
	monitorDir = 'gymRecordings'
	experienceDir = 'expMemory' #TODO-kill
	trainSummaryDir = 'summaries/train'
	playSummaryDir = 'summaries/play'
