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
from datetime import timedelta
from scipy.sparse import coo_matrix

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
        self.markovTransMatrix = []
        self.markovDimSizes = []
        self.dataBinCenters = []
        self.stateValidityMap = [] # check using np.array()

        # Stats data
        self.dataBinCounts = []
        self.dataBinEdges = []
        self.dataBin = []
        self.dataBinWidth = []
        self.dataCDF = []

    def import_dataTable(self, filename):

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
        self.rawData = pd.read_csv(filename, parse_dates=['datetime_utc_measured'], index_col='datetime_utc_measured')
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
        self.stateValidityMap = np.ones(self.numRows)

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

        for dataIx in range(0,data.shape[1]):
            if num_bins is None:
                array,edges = np.histogram(data[:,dataIx],density=True)
            else:
                array,edges = np.histogram(data[:,dataIx],bins=hist_bins[dataIx],density=True)

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
            self.dataBinCounts[dataIx] = edges
            self.dataBinCenters[dataIx] = edges[1:-1]+np.diff(edges[0:2])[0]/2

            # Subtract 1 from bin to map state to array index
            self.dataBin[dataIx] = np.digitize(data[:,dataIx],edges) - 1

        # TODO: output validation on the dataBin arrays to ensure all values are valid (not NaN, zero, etc)
        
    def computeStateValidityMap(self, maxStepSize, finiteCheck = True):
        self.stateValidityMap = pd.Series(self.rawData.index).diff() > maxStepSize

        # self.stateValidityMap = [True for i in range(self.rawData.shape[0])]
        # for idx in range(1,self.rawData.size-1):
        #     self.stateValidityMap[idx] = not(self.rawData.index[idx+1] - self.rawData.index[idx] > maxStepSize)

    def computeNdMarkovMatrix(self, forceRegen = False):
        return MarkovModel.genNdMarkovMatrix(self.dataBin, self.stateValidityMap)

    @staticmethod
    def genNdMarkovMatrix(stateVector, stateValidityMap):

        # stateVector is a dict of [1xN] vectors, each element a dimension of the time series
        nStateDim = len(stateVector)            
        dimSize = np.zeros(nStateDim)
        
        uniqueStateVals = {}
        for idx in range(nStateDim):
                
            # Pack the unique values for each dimension so we don't have to
            # keep searching for it
            uniqueStateVals[idx] = np.unique(stateVector[idx])
            
            # Store the largest state in a separate array
            dimSize[idx] = max(uniqueStateVals[idx])

        # Calculate number of unique state combinations
        nStateComb = int(np.prod(dimSize))

        # Convert to tuple for unravel index
        dimSizeT = tuple([int(i) for i in dimSize])

        # TODO: Remove this, unnecessary transformation from dict to matrix?
        stateData = np.array([stateVector[i] for i in range(len(stateVector))])

        # Represent as an ndarray, do we need this?
        stateMatrix = np.ndarray(np.shape(stateData),buffer=stateData,dtype='int')

        # Represent the state data as an array of tuples
        stateT = list(map(tuple,stateData.T))

        # Iterate through each combination of states and determine the occurrence of each
        nTimesInStateIx = np.zeros(nStateComb)

        uniqueStates = list(set(stateT))

        for sdx in uniqueStates:
            ix = np.ravel_multi_index(sdx,dimSizeT, order='C')
            nTimesInStateIx[ix] = stateT.count(sdx)

        # nTimesInStateIx = [stateT.count(np.unravel_index(sdx, dimSizeT, order='C')) for sdx in range(nStateComb)]




        nodeTxMatrix = coo_matrix((3, 4, 5), dtype=np.int32).tocsr()



        return (nodeTxMatrix,nTimesInStateIx)

    def binned_statistic(x, values, func, nbins, range):
        '''The usage is nearly the same as scipy.stats.binned_statistic''' 

        N = len(values)
        r0, r1 = range

        digitized = (float(nbins)/(r1 - r0)*(x - r0)).astype(int)
        S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))

        return [func(group) for group in np.split(S.data, S.indptr[1:-1])]

if __name__ == '__main__':

    # instantiate a MarkovModel object
    dd = MarkovModel()
    dd.import_dataTable('measLoadData_truncated.csv')
    dd.display_dataTable()
    #dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
    dd.configure_dataTable(['total_demand_kw'])
    dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')