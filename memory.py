#TODO TF & restoring
import numpy as np
import random

class ExperienceMemory(object):

    def __init__(self, state_shape ,buffer_size = 50000): # buff size = no. of exp tuples

        self.stateMem = np.empty([buffer_size,state_shape[0],state_shape[1],state_shape[2]])
        self.actionMem = np.empty([buffer_size],dtype = int)
        self.rewardMem = np.empty([buffer_size], dtype= float)
        self.doneMem = np.empty([buffer_size],dtype= bool)
        self.nxt_stateMem = np.empty([buffer_size,state_shape[0],state_shape[1],state_shape[2]])

        self.buffer_size = buffer_size
        self.isFull = False
        self.indexer = 0

    def add(self,experience):

        if self.indexer == self.buffer_size:
            self.indexer = 0
            self.isFull = True
            print("memory refill")

        self.stateMem[self.indexer] = experience[0]
        self.actionMem[self.indexer] = experience[1]
        self.rewardMem[self.indexer] = experience[2]
        self.doneMem[self.indexer] = experience[3]
        self.nxt_stateMem[self.indexer] = experience[4]

        self.indexer += 1


    def sample(self,size):

        if self.isFull == True:
            assert self.buffer_size >= size, "batch size can't be larger than memory size!"
            indexes = random.sample(range(self.buffer_size), size)

        else:
            assert self.indexer >= size, "batch size can't be larger than currently filled memory!"
            indexes = random.sample(range(self.indexer), size)

        return self.getSamples(indexes)
        #return self.stateMem[indexes], self.actionMem[indexes], self.rewardMem[indexes], self.doneMem[indexes], self.nxt_stateMem[indexes]

    def getSamples(self,indexes):

        return self.stateMem[indexes], self.actionMem[indexes], self.rewardMem[indexes], self.doneMem[indexes], self.nxt_stateMem[indexes]

    def update(self,deltas):
        raise ("Not implemented!")

    def getISW(self):
        raise("Not implemented!")

    def betaAnneal(self,s):
        raise("Not implemented!")


