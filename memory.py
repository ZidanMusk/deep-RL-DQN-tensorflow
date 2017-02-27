import numpy as np
import random

class ExperienceMemory(object):
	def __init__(self, buffer_size = 50000): # buff size = no. of exp tuples
		self.buffer = []
		self.buffer_size = buffer_size
		self.isFull = False
		self.count = 0

	def add(self,experience):
		
		if len(self.buffer) >= self.buffer_size:
			del self.buffer[0]
			self.isFull = True
	
		self.buffer.append(experience)
		self.count +=1
	
	
	def sample(self,size):
		pass

		if self.isFull == True:
			assert self.buffer_size >= size, "batch size can't be larger than memory size!"
			indexes = np.random.randint(0,self.buffer_size,size = size)

		else:
			assert self.count >= size, "batch size can't be larger than currently filled memory!"
			indexes = np.random.randint(0,self.count, size = size)
	
		return map(np.array, zip(* np.asarray(self.buffer)[indexes]))
		