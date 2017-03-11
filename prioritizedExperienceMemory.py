import math
import random
import tensorflow as tf
from sumTree import SumTree
from settings import PerSettings
from memory import ExperienceMemory


class PEM(ExperienceMemory):

    def __init__(self, state_shape ,buffer_size):
        ExperienceMemory.__init__(self,state_shape ,buffer_size)

        self.epsilon = PerSettings.epsilon
        self.alpha = PerSettings.alpha

        self.priorityTree = SumTree(self.alpha, buffer_size)

        self.betaInit = PerSettings.beta_init
        self.betaFinal = PerSettings.beta_final
        self.betaFinalAt = PerSettings.beta_finalAt

        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="beta")
        self.betaHolder = tf.placeholder(dtype=tf.float32)
        self.betaUpdater = self.beta.assign(self.betaHolder)

        with tf.variable_scope('AgentEnvSteps', reuse = True):
            self.curStep = tf.get_variable(name='agentSteps',dtype=tf.int32)

        pass
        self.impSamplingWeights = []
        self.sampledMemIndexes = []


    def betaAnneal(self, sess):
        pass
        ff = max(0, (self.betaFinal - self.betaInit) * (self.betaFinalAt - self.curStep.eval()) / self.betaFinalAt)
        bt = self.betaFinal - ff
        sess.run(self.betaUpdater, feed_dict={self.betaHolder : bt})


    def add(self,experience):

        ExperienceMemory.add(self,experience)
        #init new transitions priorities with maxPriority!
        self.priorityTree.addNew(self.priorityTree.getMaxPriority())


    def sample(self, k):

        pTotal = self.priorityTree.getSigmaPriority()
        pTot_by_k = int(pTotal // k)

        self.sampledMemIndexes = []
        self.impSamplingWeights = []

        for j in range(k):

            lower_bound = j * (pTot_by_k)
            upper_bound =  (j+1) * (pTot_by_k)
            sampledVal = random.sample(range(lower_bound,upper_bound),1)

            sampledMemIdx, sampledPriority = self.priorityTree.getSelectedLeaf(sampledVal[0])

            self.sampledMemIndexes.append(sampledMemIdx)

            assert sampledPriority !=0.0, "Can't progress with a sampled priority = ZERO!"

            sampledProb  = (sampledPriority ** self.alpha) / self.priorityTree.getSigmaPriority(withAlpha= True)

            impSampleWt = (self.buffer_size * sampledProb) ** (-1 * self.beta.eval())
            self.impSamplingWeights.append(impSampleWt)

        #normalize imp-weighted sampling
        maxISW = max(self.impSamplingWeights)
        self.impSamplingWeights[:] = [x / maxISW for x in self.impSamplingWeights]

        return self.getSamples(self.sampledMemIndexes)

    def getISW(self):

        return self.impSamplingWeights

    def update(self,deltas):

        for i,memIdx in enumerate(self.sampledMemIndexes):
            new_priority  = math.fabs(deltas[i]) + self.epsilon
            self.priorityTree.updateTree(memIdx, new_priority)
