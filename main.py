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
from IPython.display import display

class MarkovModel:

    def __init__(self):
        random.seed()
        self.randSeed = random.randint(1,pow(2,31)) # 2^31 = maxint for signed 32 bit number
        self.rawData  = []
        self.stateValidityMap = [] # check using np.array()
        self.numCols = 0
        self.numRows = 0
        self.__dataHandle = None

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


if __name__ == '__main__':

    # instantiate a MarkovModel object
    dd = MarkovModel()
    dd.import_dataTable('measLoadData_truncated.csv')
    dd.display_dataTable()
    #dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
    dd.configure_dataTable(['total_demand_kw'])
    dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')