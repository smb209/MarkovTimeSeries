
from MarkovModel import MarkovModel
from datetime import timedelta
from os.path import exists
from pandas import read_pickle

# instantiate a MarkovModel object
dd = MarkovModel()

fileName = 'measLoadData'

if exists(fileName + '.pkl'):
    dd.rawData = read_pickle(fileName + '.pkl')
else:
    dd.import_dataTable(fileName + '.csv')
    dd.rawData.to_pickle(fileName + '.pkl')

# Add the second state dimension
dd.rawData['minute'] = [60*t.hour + t.minute for t in dd.rawData.index]
dd.rawData['hour'] = [t.hour for t in dd.rawData.index]

dd.display_dataTable()
#dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
dd.configure_dataTable(['total_demand_kw','minute','hour'])
# dd.plot_dataTable('Date/Time','Load (kW)','Building Load','blue')
dd.create_state_bins([50,144,24])
dd.computeStateValidityMap(timedelta(minutes=10))
dd.computeNdMarkovMatrix(forceRegen=True)
print("Done")