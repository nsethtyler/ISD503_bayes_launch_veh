'''
Seth Tyler
ISD503 Project
Fall 2021
'''


import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import sys

# Setup to allow running a specific launch number from the command line
try:
    launches_to_analyze = sys.argv[1]
except:
    launches_to_analyze = 3

# This datafile is created by running "make_dataset_spec_launches.ipynb"
datafile = '../08-Datasets/filtered_to_'+str(launches_to_analyze)+'_launches.xlsx'
sheet = 'ForNeticaShortExcluded'

# Alternately, you can use the full datafile, but it uses all launches, all data:
#datafile = '../08-Datasets/Aggregated_SLR_Dataset_v4.xlsx'

# pull data from excel file
data = pd.read_excel(datafile, sheet_name=sheet,engine="openpyxl") #if using .xlsx

# Rename columns to shorter values so that they appear on the screen
data = data.rename(columns={'Number_of_Stages_Category':'Stages_Cat', 'Veh_Generation_Category':'Veh_Gen_Cat'})

print("\nNumber of launches analyzed:",launches_to_analyze)
print("Number of launches included in data set:", len(data.index),'\n')

# This is the author-created model of the dependencies
model = BayesianNetwork([('Mass_Cat', 'Stages_Cat'), ('Stages_Cat', 'Failure_Bool'), ('Veh_Gen_Cat', 'Failure_Bool')])

model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")#, equivalent_sample_size=10) # default equivalent_sample_size=5

# Print the CPTs
for cpd in model.get_cpds():
    print(f'CPT of {cpd.variable}:')
    print(cpd, '\n')
    print(cpd.variables,'\n')
    print(cpd.values, '\n')