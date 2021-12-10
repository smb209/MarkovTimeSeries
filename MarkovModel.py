# Description: this code obtains a Markov matrix for measured time-series data and performs resampling to obtain a
# synthesized data set to match the underlying statistical properties of the measured data.
#

import os.path
import argparse
import random

from dateutil.parser import parse
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd
from math import isclose, nan
from pandas.core.indexes.base import InvalidIndexError
from scipy.sparse import csr_matrix
from IPython.display import display
from datetime import timedelta, datetime
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
import random

class MarkovModel:

    def __init__(self):
        random.seed()
        self.randSeed = random.randint(1,pow(2,31)) # 2^31 = maxint for signed 32 bit number
        self.rawData  = []
        self.numCols = 0
        self.numRows = 0
        self.__dataHandle = None

        self.nHistBins = 50
        self.nTimesInStateIx = []

        # Markov matrix properties
        self.markovTransMatrix = None
        self.markovDimSizes = None
        self.dataBinCenters = None
        self.stateValidityMap = None # check using np.array()

        # Stats data
        self.dataBinCounts = None
        self.dataBinEdges = None
        self.dataBin = None
        self.dataBinWidth = None
        self.dataCDF = None
        self.uniqueStates = None
        self.stateCounts = None
        # Number of states for each dimension
        self.numStates = None

    def import_dataTable(self, filename, timeColumn):

        #def import_dataTable(filename, displayOption="nodisplay", plotOption="noplot"):
        # check if file name provided exists
        try:
            os.path.isfile(filename)
        except FileNotFoundError:
            return

        numDashes = 75   # number of dashes to display in header

        print("\n")
        print("-"*numDashes + "\n")
        print("Importing time-series data from {0}...".format(filename))

        # Import as measurements as dataframe
        self.rawData = pd.read_csv(filename, parse_dates=[timeColumn], index_col=timeColumn)
        self.rawData = self.rawData.to_period("min")

        # Sort the dataframe by ascending time
        self.rawData.sort_index(ascending=True, inplace=True)
        print("Data import successful...".format(filename))

    def plot_dataTable(self, xlabel='', ylabel='', title='', dcolor='red',dpi=100):

        if not ( isinstance(title, str) ):
            print("Error in plot_dataTable(): argument must be a string.")
            return

        __x = self.rawData.index
        __y = self.rawData.values

        plt.figure(figsize=(16, 5), dpi=dpi)
        plt.plot(__x, __y, color=dcolor)
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.yticks(fontsize=14, alpha=.7)
        plt.xticks(fontsize=14, alpha=.7)
        plt.show()

    def display_dataTable(self, numLines=10):

        numDashes = 75  # number of dashes to display in header

        if not ( isinstance(numLines, int) and  numLines > 0 ):
            print("Error in display_dataTable(): argument must be a non-zero integer.")
            return

        # display the first N lines of data
        display(self.rawData.head(numLines))
        print(".\n" * 3)
        print("[ " + str(self.rawData.shape[0]) + " rows x " + str(self.rawData.columns.size) + " columns ]")


    def configure_dataTable(self, columnsToProcess):

        if not (all(isinstance(item, str) for item in columnsToProcess)):
            print("Error in MarkovModel.importTimeTable(): second argument must be a string or list of strings.")
            return

        if not (all(item in self.rawData.columns for item in columnsToProcess )):
            print("Error in MarkovModel.importTimeTable(): not all columns requested are in data file.")
            return

        print("Configuring data table...")

        # save down-selected columns to rawData
        self.rawData = self.rawData[columnsToProcess]

        # save attributes of down-selected rawData
        self.numRows = self.rawData.shape[0]
        self.numCols = self.rawData.shape[1]
        print("Data table configured with ..." + str(self.numRows) +" rows (time-steps) and " + str(self.numCols) +" data column(s)." )

        # set state validity map to true (=1's)
        self.stateValidityMap = np.ones(self.numRows, dtype=bool)

    def create_state_bins(self, num_bins = None):
        data = np.array(self.rawData)

        hist_bins = np.array(num_bins)
        
        if np.size(hist_bins) == 1:
            hist_bins = np.tile(hist_bins,[1,data.shape[1]])
        elif np.size(hist_bins) > data.shape[1]:
            raise ValueError('Hist bins provided is greater than the number of data columns')
        
        self.dataBinCounts = {}
        dataBinEdges = {}
        self.dataCDF = {}
        self.dataBin = {}
        self.dataBinCenters = {}
        self.dataBinEdges = {}
        self.dataBinWidth = {}
        self.numStates = np.zeros(data.shape[1],dtype='int')
        for dataIx in range(0,data.shape[1]):
            if num_bins is None:
                array,edges = np.histogram(data[:,dataIx],density=True)
            else:
                array,edges = np.histogram(data[:,dataIx],bins=hist_bins[dataIx],density=True)

            # Store the bins and edges
            self.dataBinCounts[dataIx] = array
            self.dataBinEdges[dataIx] = edges
            self.dataBinWidth[dataIx] = np.diff(edges[0:2])[0]/2

            # Store the number of states for the dataIx dimension for later use
            self.numStates[dataIx] = len(array)

            # Normalize CDF so there are no duplicate entries
            dataCdf = np.cumsum(array) # By definition must be sorted
            dataCdf_tmp = np.zeros(np.size(dataCdf))
            for i in range(0,len(dataCdf)):
                if i == 0:
                    dataCdf_tmp[i] = dataCdf[i]
                elif isclose(dataCdf[i-1],dataCdf[i]):
                    dataCdf_tmp[i] = dataCdf[i]+0.001*i # Bump by a small amount
                else:
                    dataCdf_tmp[i] = dataCdf[i]

            self.dataCDF[dataIx] = dataCdf_tmp
            self.dataBinCenters[dataIx] = edges[1:]+np.diff(edges[0:2])[0]/2
            
            # Discard the first bin edge to get the correct number of states. Anything < the min will be included in state 0
            self.dataBin[dataIx] = np.searchsorted(edges[1:],data[:,dataIx], side='left')

        # TODO: output validation on the dataBin arrays to ensure all values are valid (not NaN, zero, etc)
        
    def computeStateValidityMap(self, maxStepSize, finiteCheck = True):
        self.stateValidityMap = np.array(pd.Series(self.rawData.index).diff() <= maxStepSize)

        # Set the last element to invalid since there is no next state
        self.stateValidityMap[-1] = False

        # self.stateValidityMap = [True for i in range(self.rawData.shape[0])]
        # for idx in range(1,self.rawData.size-1):
        #     self.stateValidityMap[idx] = not(self.rawData.index[idx+1] - self.rawData.index[idx] > maxStepSize)

    def computeNdMarkovMatrix(self, forceRegen = False):
        if self.markovTransMatrix is None or forceRegen:
            self.markovTransMatrix,self.nTimesInStateIx = MarkovModel.genNdMarkovMatrix(self.dataBin, tuple(self.numStates), self.stateValidityMap)


    def removeTerminalStates(self):
        pass

    @staticmethod
    def plotTransitionMatrix(mat):
        fig,ax = plt.subplots(figsize=(8, 4), dpi= 80, facecolor='w', edgecolor='k')
        fig.suptitle('Markov Transition Matrix')
        
        # Convert to subscriptable matrix
        matx = mat.tocsr()

        # prepare x and y for scatter plot
        plot_list = []
        nz = np.nonzero(matx)
        for rows,cols in zip(nz[0],nz[1]):
            plot_list.append([cols,rows,matx[rows,cols]])
        plot_list = np.array(plot_list)

        # scatter plot with color bar, with rows on y axis
        plt.scatter(plot_list[:,0],plot_list[:,1],c=plot_list[:,2], s=50)
        cb = plt.colorbar()

        # full range for x and y axes
        plt.xlim(0,matx.shape[1])
        plt.ylim(0,matx.shape[0])
        # invert y axis to make it similar to imshow
        plt.gca().invert_yaxis()


    @staticmethod
    def genNdMarkovMatrix(stateVector, dimSize, stateValidityMap):
        # stateVector is a dict of [1xN] vectors, each element a dimension of the time series
        
        if not type(dimSize) == tuple:
            raise ValueError('dim size must be provided as a tuple')

        uniqueStateVals = {}
        for udx in range(len(dimSize)):
                
            # Pack the unique values for each dimension so we don't have to
            # keep searching for it
            uniqueStateVals[udx] = np.unique(stateVector[udx])

        # Calculate number of unique state combinations
        nStateComb = int(np.prod(dimSize))

        # TODO: Remove this, unnecessary transformation from dict to matrix?
        stateData = np.array([stateVector[i] for i in range(len(stateVector))]).T

        # Count the number of each unique state
        uStates, stCounts = np.unique(stateData,axis=0,return_counts=True)

        # Iterate through each combination of states and determine the occurrence of each
        nTimesInStateIx = np.zeros(nStateComb,dtype='int32')

        # Look at the number of samples of this bin relative to others.
        # If there are significantly more, provide an allocation and
        # randomly discard any beyond that. This may or may not be used
        # depending on the density of each state occurrence
        allocation = np.floor(stCounts.std()*3)

        # Initialize a spare matrix to store the node transitions
        # TODO: Determine if float16 is okay for the probabilities
        nodeTxMatrix = dok_matrix((nStateComb,nStateComb), dtype=np.float32)

        ## Use np ravel/unravel index to map ND -> 1D
        # linIx = np.ravel_multi_index((4,1,2),dimSize, order='C')
        # tupleIx = np.unravel_index(5513, dimSize, order='C')

        for udx, uState in enumerate(uStates):

            #if udx > 500:
            #    break

            # Find linear index of source state
            srcIx = np.ravel_multi_index(uState,dimSize, order='C')

            # Find logical array where the unique state occurs
            boolVec = (stateData == uState).all(axis=1)

            # AND it with the state validity map to remove any invalid entries
            stateIx = np.logical_and(np.array(stateValidityMap),boolVec)

            ## Use np.nonzero(...)[0] to find indices of nonzero elements
            # stateIxs = np.nonzero(stateIx)[0]

            # If there are no valid states remaining, skip this iteration
            if np.sum(stateIx) == 0:
                continue

            # TODO: add reduction of over allocated states

            # Shift indicies by 1 to find the k+1 state
            iPlusIdxs = np.insert(stateIx[:-1],[0],False,axis=0)

            # Find the next state, counts, and transition probability
            nextStates,nextCounts = np.unique(stateData[iPlusIdxs],axis=0,return_counts=True)
            nextProbs = nextCounts / np.sum(nextCounts)

            ##nodeTxMatrix[srcIx] = {}
            for ndx,nextState in enumerate(nextStates):
                destIx = np.ravel_multi_index(nextState,dimSize, order='C')
                nodeTxMatrix[srcIx,destIx] = nextProbs[ndx]

        return (nodeTxMatrix.tocsr(),nTimesInStateIx)

    
    # TODO: Make static method?
    # @staticmethod
    def removeTerminalStates(self, maxIterations = 50):
        count = 0
        while True:
            # Find the populated elements of the sparse transition matrix
            rowIx, colIx = self.markovTransMatrix.nonzero()

            orphanI = set(rowIx).difference(set(colIx))
            orphanJ = set(colIx).difference(set(rowIx))

            if len(orphanI) == 0 and len(orphanJ) == 0:
                break

            for j in orphanJ:
                self.removeDeadEndPaths(j,direction='back')
                self.removeDeadEndPaths(j,direction='forward')

            for i in orphanI:
                self.removeDeadEndPaths(i,direction='forward')
                self.removeDeadEndPaths(i,direction='back')

            # TODO: Prevent infinte loop
            count += 1
            if count > maxIterations:
                Warning("Iteration limit reached in findTerminalStates, exiting")
                break

        print("Removed terminal nodes after {} iterations".format(count))

        # # Loop through each unique state, find the linear index, and extract the transition column
        # for state in uStates:
        #     linState = np.ravel_multi_index(state,tuple(self.numStates), order='C')
        #     exitNodes = self.markovTransMatrix[linState,:].nonzero()

        #     # If there are no exit nodes, its a terminal state
        #     if len(exitNodes) == 1:
        #         termStates.append(state)

    def removeDeadEndPaths(self, linIx, direction = "back"):

        if direction == "forward":
            # TODO: This could probably be simplified by looking for any elements with a P=1.0 transition probability
            destNodes = self.markovTransMatrix[linIx,:].nonzero()[1]
            
            if len(destNodes) == 1:
                # Move forward
                self.removeDeadEndPaths(destNodes[0],direction)

            elif len(destNodes) > 1:
                # We found a node with multiple destinations, Stop
                count = 4
                pass

            else:
                # We reached a dead end
                count = 4
                pass

            if len(destNodes) > 0:
                # If we found anything, remove it
                # Remove the node
                self.markovTransMatrix[linIx,destNodes] = 0
                self.markovTransMatrix.eliminate_zeros()
                

        elif direction == "back":

            # Find any nodes where this terminal node is the destination
            sourceNodes = self.markovTransMatrix[:,linIx].nonzero()[0]

            if len(sourceNodes) == 0:
                # TODO: This should always be a no-op going backwards, verify if its needed going forwards
                nonZeroIxs = self.markovTransMatrix[linIx,:].nonzero()[1]
                self.markovTransMatrix[linIx,nonZeroIxs] = 0

                # Remove the new zeros from the sparse matrix storage
                # TODO: Probably inefficient, clean up later
                self.markovTransMatrix.eliminate_zeros()

            else:

                for sourceNode in sourceNodes:
                    # Check to see if nodes in sourceNodes have other outgoing paths
                    #  If not, remove node and rescale probabilities
                    nodes = self.markovTransMatrix[sourceNode,:].nonzero()
                    destNodes = nodes[1]

                    if len(destNodes) == 1:
                        # There is a single path, keep searching backwards
                        self.removeDeadEndPaths(sourceNode,direction)
                        
                        nonZeroIxs = self.markovTransMatrix[linIx,:].nonzero()[1]
                        self.markovTransMatrix[linIx,nonZeroIxs] = 0

                        # Remove the new zeros from the sparse matrix storage
                        # TODO: Consolidate the sparsity cleanup
                        self.markovTransMatrix.eliminate_zeros()

                    else:
                        # Multiple nodes found, remove the terminal path (to linIx) and 
                        # rescale the remaining probabilties to sum to 1
                        ixsToRemove = (destNodes == linIx)

                        # Find the total probability of nodes being removed
                        probabilityToRedistribute = self.markovTransMatrix[sourceNode,destNodes[ixsToRemove]].sum()

                        # Divide that across remaining valid nodes
                        amountToAddToOtherNodes = probabilityToRedistribute/(~ixsToRemove).sum()

                        # Temp variable and sparse assignment
                        tmpToKeep = self.markovTransMatrix[sourceNode,destNodes[~ixsToRemove]].todense()
                        self.markovTransMatrix[sourceNode,destNodes[~ixsToRemove]] = tmpToKeep+amountToAddToOtherNodes

                        # Zero the removed nodes
                        self.markovTransMatrix[sourceNode,destNodes[ixsToRemove]] = 0
                        self.markovTransMatrix.eliminate_zeros()
            pass
        else:
            raise ValueError("direction must be either 'back' or 'forward'.")
            pass

        ## Use np ravel/unravel index to map ND -> 1D
        # linIx = np.ravel_multi_index((4,1,2),dimSizeT, order='C')
        # tupleIx = np.unravel_index(5513, self.dimSizeT, order='C')

    @staticmethod
    def getUniqueStatesAndCounts(dataBin):
        # Find the most frequent state in the input data to use as the initial state
        uStates, stCounts = np.unique(np.array([dataBin[i] for i in range(len(dataBin))]).T,axis=0,return_counts=True)
        return (uStates, stCounts)

    def binned_statistic(x, values, func, nbins, range):
        '''The usage is nearly the same as scipy.stats.binned_statistic''' 

        N = len(values)
        r0, r1 = range

        digitized = (float(nbins)/(r1 - r0)*(x - r0)).astype(int)
        S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))

        return [func(group) for group in np.split(S.data, S.indptr[1:-1])]

    def genSampleData(self,numSamples,
        startingTime=datetime.strptime('2020-12-1 12:00:00','%Y-%m-%d %H:%M:%S'),
        stepSizeMinutes=timedelta(minutes=10),
        initSample = None,
        initialMarkovStates = None):
        
        # If the random seed was set, seed the RNG
        if self.randSeed is not None:
            random.seed(self.randSeed)

        if initialMarkovStates is None:
            # Find the most frequent state in the input data to use as the initial state
            uStates, stCounts = self.getUniqueStatesAndCounts(self.dataBin) # np.unique(np.array([self.dataBin[i] for i in range(len(self.dataBin))]).T,axis=0,return_counts=True)
            initialMarkovStates = uStates[np.argmax(stCounts)]

        # Generate the random walk using the linear state index    
        linState = np.ravel_multi_index(initialMarkovStates,tuple(self.numStates), order='C')

        # Preallocate output matrix
        nStates = len(self.numStates)
        dataSamp = np.zeros((numSamples,nStates),dtype='float32')

        ## Use np ravel/unravel index to map ND -> 1D
        # linIx = np.ravel_multi_index((4,1,2),dimSizeT, order='C')
        # tupleIx = np.unravel_index(5513, dimSizeT, order='C')

        self.markovTransMatrix = self.markovTransMatrix.tocsr()

        for sdx in range(numSamples):
            
            # Decode the linear state into i,j,... states
            tupleIx = np.unravel_index(linState, tuple(self.numStates), order='C')

            # Convert the state back to unit values
            dataSamp[sdx,:] = [self.dataBinCenters[ix][tupleIx[ix]] for ix in range(nStates)]

            # Extract transition probabilities from the transition matrix and compute the CDF to inverse sample
            # Since the markov matrix is stored sparsely, we need to find the nonzero elements in the row
            rows,cols = self.markovTransMatrix[linState,:].nonzero()

            if len(cols) == 0:
                raise ValueError("Transition matrix has no exit nodes from position {}".format(tupleIx))

            cdf = np.cumsum(self.markovTransMatrix[linState,cols].todense()).flat

            # Create a uniform random sample
            rand = random.uniform(0, 1)

            # Look for the first element in the CDF that is greater than
            item = next((x for x in cdf if x > rand), None)

            # Check the iterator coordinates to see where the match was found 
            # 
            # Edge case: If no match was found assume it is greater than the CDF
            # This is possible due to round off of floats where the max value of
            # the CDF could be 0.99999993 or something similar
            if item is not None:
                nextState = cdf.coords[1]
            else:
                nextState = len(cols)

            # Set the next state
            linState = cols[nextState-1]

        # Return the new array of samples
        return dataSamp

if __name__ == '__main__':

    # instantiate a MarkovModel object
    dd = MarkovModel()
    dd.import_dataTable('measLoadData_truncated.csv')
    dd.display_dataTable()
    #dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
    dd.configure_dataTable(['total_demand_kw'])
    dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')