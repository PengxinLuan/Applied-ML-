import numpy as np
from collections import Counter
from ordered_set import OrderedSet
from graphviz import Digraph
from sklearn import tree, metrics, datasets
from ordered_set import OrderedSet


class ID3RegressionTreePredictor :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2, maxDepth = 100, stopMSE = 0.0) :

        self.__nodeCounter = -1
        
        self.__dot = Digraph()

        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit
        self.__maxDepth = maxDepth
        self.__stopMSE = stopMSE

        self.__numOfAttributes = 0
        self.__attributes = None
        self.__target = None
        self.__data = None

        self.__tree = None

    def newID3Node(self):
        self.__nodeCounter += 1
        return {'id': self.__nodeCounter, 'splitValue': None, 'nextSplitAttribute': None, 'mse': None, 'samples': None,
                         'avgValue': None, 'nodes': None}


    def addNodeToGraph(self, node, parentid):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != None):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        #print(nodeString)

        return

    
    def makeDotData(self) :
        return self.__dot

    
    #Calculating MSE for one set
    def calcMSE(self, dataIDXs) :
        numOfSamples = len(dataIDXs)
        mse = 0.0
        avg = 0.0
        
        splitTargets = [self.__target[i] for i in dataIDXs]
        
        if( numOfSamples !=0) :
            avg = np.mean(splitTargets)
            mse = np.sum((splitTargets-avg)**2)

        return mse, avg

    # Calculating MSEs for all "new" sets, this method can be part of findSplitAttr 
    def calcOverallMSE(self, attribute, dataIDXs) :
        mse_s = dict()
        
        averages = dict()

        nChildren = dict()
        splitDataIDXs = dict()
        dataIDXs_copy = dataIDXs.copy()

        nParent = len(dataIDXs)


        for val in self.__attributes[attribute] :
            nChildren[val] = 0
            splitDataIDXs[val] = []
            #print(val)
            for sampleIDX in dataIDXs_copy:
                # Super important to get this right, otherwise the wrong attribute is used
                if( self.__data[sampleIDX][list(self.__attributes.keys()).index(attribute)] == val) :
                    splitDataIDXs[val].append(sampleIDX)
                    nChildren[val] += 1

            for i in range(len(splitDataIDXs[val])) :
                dataIDXs_copy.remove(splitDataIDXs[val][i])
            #print( idx)
            mse_s[val] = 0.0
            mse_s[val], averages[val] = self.calcMSE(splitDataIDXs[val])
            
        overallMSE = np.sum(list(mse_s.values()))

        return overallMSE, mse_s, averages, splitDataIDXs
    

    # Finding the best split attribute
    def findSplitAttr(self, attributes, dataIDXs) :

        minMSE = float("inf")
            
        splitAttr = ''
        splitMSEs = {}
        splitDataIDXsFinal = {}
        splitAveragesFinal = {}

        for attr in attributes :
            
            overallMSE, subMSEs, averages, splitDataIDXs = self.calcOverallMSE(attr, dataIDXs)
            
            if overallMSE < minMSE:
                splitAttr = attr
                #print(splitAttr)
                minMSE = overallMSE
                splitMSEs = subMSEs
                splitDataIDXsFinal = splitDataIDXs
                splitAveragesFinal = averages
            

        return minMSE, splitAttr, splitMSEs, splitAveragesFinal, splitDataIDXsFinal


    # the starting point for fitting the tree
    def fit(self, data, target, attributes, attributesToTest):

        self.__numOfAttributes = len(attributes)
        self.__attributes = attributes
        self.__data = data
        self.__target = target

        
        dataIDXs = {j for j in range(len(data))}

        mse, avg = self.calcMSE(dataIDXs)
        
        self.__tree = self.fit_rek( 0, None, '-', attributesToTest, mse, avg, dataIDXs)

        return self.__tree


    # the recursive tree fitting method
    def fit_rek(self, depth, parentID, splitVal, attributesToTest, mse, avg, dataIDXs) :

        root = self.newID3Node()
        
        root.update({'splitValue':splitVal, 'mse': mse, 'samples': len(dataIDXs)})
        currentDepth = depth
               
        if (currentDepth == self.__maxDepth or mse <= self.__stopMSE or len(attributesToTest) == 0 or len(dataIDXs) < self.__minSamplesSplit):
            root.update({'avgValue':avg})
            self.addNodeToGraph(root, parentID)
            return root

        minMSE, splitAttr, splitMSEs, splitAverages, splitDataIDXs = self.findSplitAttr(attributesToTest, dataIDXs)


        root.update({'nextSplitAttribute': splitAttr, 'nodes': {}})
        self.addNodeToGraph(root, parentID)

        attributesToTestCopy = OrderedSet(attributesToTest)
        attributesToTestCopy.discard(splitAttr)

        #print(splitAttr, splitDataIDXs)

        for val in self.__attributes[splitAttr] :
            #print("testing " + str(splitAttr) + " = " + str(val))
            if( len(splitDataIDXs[val]) < self.__minSamplesLeaf) :
                root['nodes'][val] = self.newID3Node()
                root['nodes'][val].update({'splitValue':val, 'samples': len(splitDataIDXs[val]), 'avgValue': splitAverages[val]})
                self.addNodeToGraph(root['nodes'][val], root['id'])
                print("leaf, not enough samples, setting node-value = " + str(splitAverages[val]))
                
            else :
                root['nodes'][val] = self.fit_rek( currentDepth+1, root['id'], val, attributesToTestCopy, splitMSEs[val], splitAverages[val], splitDataIDXs[val])

        return root

    # Doing a prediction for a data set 'data' (starting method for the recursive tree traversal)
    def predict(self, data) :
        return [self.predict_rek(data[i], self.__tree) for i in range(len(data))]

    # Recursively traverse the tree to find the value for the sample 'sample'
    def predict_rek(self, sample, node) :

        if(node['avgValue'] != None) :
            return node['avgValue']

        attr = node['nextSplitAttribute']
        dataIDX = list(self.__attributes.keys()).index(attr)
        val = sample[dataIDX]
        next = node['nodes'][val]

        return self.predict_rek( sample, next)
    
