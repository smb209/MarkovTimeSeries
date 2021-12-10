
from MarkovModel import MarkovModel
from datetime import timedelta
from os.path import exists
from pandas import read_pickle
import cProfile, pstats, io
from pstats import SortKey
import matplotlib.pylab as plt
import scipy.sparse as sparse
import numpy as np

# instantiate a MarkovModel object
dd = MarkovModel()

# fileName = 'measLoadData_retimed'
# dataColumns = ['total_demand_kw','minute']
# timeColumn = 'datetime_utc_measured'
# stateBins = [50,144]

fileName = '197'
dataColumns = ['value','minute','weekday']
timeColumn = 'dttm_utc'
stateBins = [50,144,7]

if exists(fileName + '.pkl'):
    dd.rawData = read_pickle(fileName + '.pkl')
else:
    dd.import_dataTable(fileName + '.csv',timeColumn)
    dd.rawData.to_pickle(fileName + '.pkl')

# Add the second state dimension
dd.rawData['minute'] = [60*t.hour + t.minute for t in dd.rawData.index]
dd.rawData['hour'] = [t.hour for t in dd.rawData.index]
dd.rawData['weekday'] = [t.weekday for t in dd.rawData.index]

dd.display_dataTable()
#dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
dd.configure_dataTable(dataColumns)
# dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
dd.create_state_bins(stateBins)
dd.computeStateValidityMap(timedelta(minutes=10))

# Process the data to compute the transition matrix
dd.computeNdMarkovMatrix(forceRegen=True)
MarkovModel.plotTransitionMatrix(dd.markovTransMatrix)

# Remove any terminal states so the generation process doesn't get stuck
termStates = dd.removeTerminalStates()

# Test an edge case where there is no state
# newData = dd.genSampleData(5000,initialMarkovStates=(0,0))

newData = dd.genSampleData(5000)

# Create some plots
plt.close('all')
histBins = 25

# Plot the original dataset
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), dpi=100)
fig1.suptitle('Original Dataset')
dd.rawData.value.plot(ax=ax1)
dd.rawData.value.hist(bins=histBins,ax=ax2,density=True)

# Plot the new dataset
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), dpi=100)
fig2.suptitle('Generated Dataset')
ax1.plot(np.matrix(newData)[:,0])
ax2.hist(np.matrix(newData)[:,0],bins=histBins,density=True)

fig1.show()
fig2.show()
plt.show()

print("Done")



## PROFILING CODE
# pr = cProfile.Profile()
# pr.enable()
# dd.computeNdMarkovMatrix(forceRegen=True)
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())