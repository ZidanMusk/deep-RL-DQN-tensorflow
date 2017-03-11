#sum tree data structure (every node is the sum of its children, with the priorities as the leaf nodes)

class SumTree(object):

    def __init__(self,alpha,expSize = 50000):

        self.treeSize = 2 * expSize -1
        self.tree = [0.0] * self.treeSize
        self.propagatedDelta = 0
        self.indexer = 0
        self.memSize = expSize
        self.initLeafIndex = expSize -1
        self.alpha = alpha


    def getSigmaPriority(self,withAlpha  = False):

        if withAlpha:
            summy = 0.0
            for x in self.tree[self.initLeafIndex:]:
                summy += (x ** self.alpha)
            return summy

        else:
            return self.tree[0]


    def getMaxPriority(self):

        maxPriority = max(self.tree[self.initLeafIndex:])
        if maxPriority == 0.:
            maxPriority = 1.0

        return maxPriority


    def updateTree(self,fuzzy_idx, new_priority, pure_leafIdx = False):

        leaf_num = fuzzy_idx

        if not pure_leafIdx:  # map mem idx to actual leaf node idx
            leaf_num += self.initLeafIndex

        self.propagatedDelta = new_priority - self.tree[leaf_num]
        self.tree[leaf_num] = new_priority

        self._upwardPropagation(leaf_num)


    def addNew(self,priority):

        addAt =  self.initLeafIndex + self.indexer

        self.indexer += 1
        if self.indexer == self.memSize:
            self.indexer = 0

        self.updateTree(addAt, priority, True) #memorty insertion must be similar


    def getSelectedLeaf(self, sampledValue):
        return self._retrieve(0, sampledValue)


    def _upwardPropagation(self,child_index):

        parent_index = (child_index -1 ) // 2
        self.tree[parent_index] += self.propagatedDelta

        if parent_index != 0: #not root...recall
            self._upwardPropagation(parent_index)


    #implicit probability
    def _retrieve(self,parent_index,value):
        #compute children indexes
        left_child_index  = parent_index * 2 + 1
        right_child_index = parent_index * 2 + 2

        if left_child_index >= self.treeSize:
            return (parent_index - self.memSize + 1), self.tree[parent_index] #corresponding memory index and priority value

        if (value <= self.tree[left_child_index] and self.tree[left_child_index] != 0.):
            return self._retrieve(left_child_index,value)
        else:
            return self._retrieve(right_child_index, (value - self.tree[left_child_index]) )
