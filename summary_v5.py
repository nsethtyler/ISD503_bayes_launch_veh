'''
Seth Tyler
ISD503 Project
Fall 2021
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
from scipy.special import gamma, btdtri
import scipy.stats as ss
import os
import sys
import matplotlib.dates as mdates
import datetime as dt


verbose = 0
save_images = 0
plot_all = 1

dividerline = '-' * 75
launch_cutoff = 49

michigan_blue = '#00274C'

plt.rcParams['figure.figsize'] = (10,6)

datafile = '../08-Datasets/Aggregated_SLR_Dataset_v4.xlsx'
datatab = 'All'

# Import data for evaluation
df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")
df.set_index('Date')

df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])

plot_mfctr_summary = 0

# By-manufacturer Summmary
if plot_mfctr_summary or plot_all:
    print('List of Launch Attempts by Manufacturer')
    print(dividerline)
    print(df['Mfctr_DID'].value_counts())

    Manufacturers=df['Mfctr_DID'].unique()
    Manufacturers.sort()
    for mftr in Manufacturers:
        if verbose: print(mftr)
        rslt_df = df.loc[df['Mfctr_DID']==mftr]
        if (rslt_df['Success_Bool'].sum() + rslt_df['Failure_Bool'].sum() > launch_cutoff):
        	if verbose: print(mftr,'exceeds')
        	plt.plot(rslt_df['Date'],rslt_df['Success_Bool'].cumsum(), label=mftr)
        if verbose:
            print("Successful launches:", rslt_df['Success_Bool'].sum())
            print("Failed launches:", rslt_df['Failure_Bool'].sum())
            print("Frequentist Failure Rate:", round(rslt_df['Failure_Bool'].sum()/(rslt_df['Failure_Bool'].sum()+rslt_df['Success_Bool'].sum()),3))
            print("\n")
    plt.legend()
    title_string = 'Launch Successes by Manufacturer \n(if total attempts>' + str(launch_cutoff) +')'
    plt.title(title_string)
    plt.grid()
    plt.show()



# By-region Summmary (Manufacture)

plot_region_summary = 0
if plot_region_summary or plot_all:
    print('List of Launch Attempts by Mftr Region')
    print(dividerline)
    print(df['Veh_Mfctr_Region'].value_counts())

    Mfctr_Regions=df['Veh_Mfctr_Region'].unique()
    # Mfctr_Regions.sort()
    for mfctr_region in Mfctr_Regions:
        rslt_df = pd.DataFrame()
        if verbose: print(mfctr_region)
        rslt_df = df.loc[df['Veh_Mfctr_Region']==mfctr_region]
        if verbose: print(mfctr_region,'exceeds')
        plt.plot(rslt_df['Date'],rslt_df['Success_Bool'].cumsum(), label=mfctr_region)
        if verbose:
            print("Successful launches:", rslt_df['Success_Bool'].sum())
            print("Failed launches:", rslt_df['Failure_Bool'].sum())
            print("Frequentist Failure Rate:", round(rslt_df['Failure_Bool'].sum()/(rslt_df['Failure_Bool'].sum()+rslt_df['Success_Bool'].sum()),3))
            print("\n")
    plt.legend()
    title_string = 'Launch Successes by Manufacture Region'
    plt.title(title_string)
    plt.grid()
    plt.show()


# By-region Summmary (Launch)

plot_regions = 0
if plot_regions or plot_all:
    print('List of Launch Attempts by Launch Region')
    print(dividerline)
    print(df['Launch_Country'].value_counts())


    Launch_Regions=df['Launch_Country'].unique()
    for launch_region in Launch_Regions:
        rslt_df = pd.DataFrame()
        if verbose: print(launch_region)
        rslt_df = df.loc[df['Launch_Country']==launch_region]
        if verbose: print(launch_region,'exceeds')
        plt.plot(rslt_df['Date'],rslt_df['Success_Bool'].cumsum(), label=launch_region)
        if verbose:
            print("Successful launches:", rslt_df['Success_Bool'].sum())
            print("Failed launches:", rslt_df['Failure_Bool'].sum())
            print("Frequentist Failure Rate:", round(rslt_df['Failure_Bool'].sum()/(rslt_df['Failure_Bool'].sum()+rslt_df['Success_Bool'].sum()),3))
            print("\n")
    plt.legend()
    title_string = 'Launch Successes by Launch Region'
    plt.title(title_string)
    plt.grid()
    plt.show()


# This section is to plot by-vehicle

plot_by_vehicle = 0
if plot_by_vehicle or plot_all:
    datatab = 'Active_New_Veh_Launches'

    # Import data for evaluation
    df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")
    df.set_index('Date')

    # Adjust data
    df['Date'] = pd.to_datetime(df['Date'])


    plt.figure()
    cutoff_value = 15.0

    for column in df.iloc[:,5:]:
        print('Checking ',column)
        if df.max()[column] > cutoff_value:
            plot_darkness = 0.2
            plot_color = 'gray'
            # plt.plot(df['Date'],df[column],color=plot_color,alpha=plot_darkness)
        else:
            plot_darkness = 0.7
            plt.plot(df['Date'],df[column],alpha=plot_darkness, label=column)

        
    # plt.legend(loc="upper left")
    plt.grid()
    # plt.xlim([80,100])
    plt.title('Total Launches per Vehicle')
    # plt.legend()
    plt.show()




#This section is to plot the overall launch count over time

plot_launch_counts = 0
if plot_launch_counts or plot_all:
    datatab = 'Metrics'

    # Import data for evaluation
    df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")
    df.set_index('Year')

    # Adjust data
    # df['Date'] = pd.to_datetime(df['Date'])

    fig, ax = plt.subplots()
    # by_year = sns.barplot(x='Year', y='Annual Total Launches', data = df, color ='#00274C') #Michigan Blue
    ax.bar(df['Year'],df['Annual Total Launches'],color =michigan_blue)
    plt.title('Total Launch Attempts Per Year (1998-2020)')

    fig.autofmt_xdate()
    plt.grid()
    if save_images:
        plt.savefig('Total_Launches_Per_Year.png',dpi=300, transparent=False, bbox_inches='tight')
    else:
        plt.show()



#This section is to plot the overall launch count by de-identified company

plot_launch_mftr = 0
if plot_launch_mftr or plot_all:
    datatab = 'Launch_Shares_Mftr'

    # Import data for evaluation
    df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")
    df.set_index('Launch Shares - DeID\'d MFR')

    # Adjust data
    # df['Date'] = pd.to_datetime(df['Date'])

    fig, ax = plt.subplots()
    # by_year = sns.barplot(x='Year', y='Annual Total Launches', data = df, color ='#00274C') #Michigan Blue
    ax.bar(df['Launch Shares - DeID\'d MFR'],df['Launch Count - static'],color =michigan_blue)
    plt.title('Total Launch Attempts Per Manufacturer (1998-2020)')

    ax.tick_params(axis='x', labelsize=8)
    plt.xticks(rotation=90)
    plt.grid(alpha=0.3)

    if save_images:
        plt.savefig('Total_Launches_Per_Mftr.png',dpi=300, transparent=False, bbox_inches='tight')
    else:
        plt.show()


# Plot manufacturer years active

plot_mftr_history = 0
if plot_mftr_history or plot_all:
    datatab = 'All'
    df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")

    plot_df = pd.DataFrame(columns=['mftr','start','end','duration'])
    fig, ax = plt.subplots()
    y_pos = 0
    Manufacturers=df['Mfctr_DID'].unique()
    Manufacturers.sort()
    for mftr in Manufacturers:
        if verbose: print(mftr)
        rslt_df = df.loc[df['Mfctr_DID']==mftr]
        rslt_df.set_index('Date')
        start = rslt_df['Date'].min()
        end = rslt_df['Date'].max()
        duration = (end-start)/np.timedelta64(1,'D')
        # print(mftr, start, end, duration)
        plot_df = plot_df.append({'mftr': mftr, 'start': start, 'end':end ,'duration':duration}, ignore_index=True)

    plot_df = plot_df.sort_values(by=['start'])

    mftr = []
    start = []
    end = []
    duration = []

    for index, row in plot_df.iterrows():
        mftr.append(row['mftr'])
        start.append(row['start'])
        end.append(row['end'])
        duration.append(float(row['duration']))
        
        y_total_count = plot_df.shape[0]
    y_pos = np.arange(len(mftr))
    
    plot_hbar = ax.barh(y_total_count-y_pos, duration, left=start, fill=False)#, color=michigan_blue)
    ax.bar_label(plot_hbar, labels=mftr, label_type='center', color=michigan_blue, fontsize=7)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.grid(True)

    formatter = mdates.DateFormatter("%Y-%m-%d")

    ax.xaxis.set_major_formatter(formatter)

    locator = mdates.MonthLocator(interval=12)

    ax.xaxis.set_major_locator(locator)
    ax.tick_params(labelrotation=35)
    ax.set_xlim([dt.datetime(1998,1,1), dt.datetime(2021,1,1)])
    plt.title('Manufacturer Years Active')
    plt.tight_layout()
    plt.show()
        


# Plot identified total launches per manufacturer - improvement on previous plot

plot_mftr_perf = 1
if plot_mftr_perf or plot_all:
    datatab = 'All'
    df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")

    rslt_df = pd.DataFrame()
    plot_df = pd.DataFrame()
    
    plot_df = pd.DataFrame(columns=['mftr','launches','successes','failures'])
    fig, ax = plt.subplots()
    y_pos = 0
    
    mftr_list = []
    successes = []
    failures = []
    launches = []

    Manufacturers=df['Mfctr_DID'].unique()
    Manufacturers.sort()
    for mftr in Manufacturers:
        if verbose: print(mftr)
        rslt_df = df.loc[df['Mfctr_DID']==mftr]
        mftr_list.append(mftr)

    
        rslt_df.set_index('Date')
        successes = max(rslt_df['Success_Bool'].cumsum())
        failures = max(rslt_df['Failure_Bool'].cumsum())
        plot_df = plot_df.append({'mftr': mftr,'successes':successes ,'failures':failures}, ignore_index=True)


    plot_df['launches'] = plot_df['successes'] + plot_df['failures']

    plot_df = plot_df.sort_values('launches', ascending=True)

    plot_hbar = ax.barh(plot_df['mftr'],plot_df['launches'], fill=False)
    ax.bar_label(plot_hbar, labels=plot_df['mftr'], label_type='center', color=michigan_blue, fontsize=7)

    ax.tick_params(labelrotation=0)
    ax.yaxis.set_ticklabels([])
    plt.title('Total Launch Attempts Per Manufacturer (1998-2020)')
    plt.tight_layout()
    # for index, row in plot_df.iterrows():
    #     print(index)
    #     ax.text(index,row['launches'],row['mftr'], ha='center',fontsize=7)
    ax.xaxis.grid(True)
    plt.show()
        
    plt.show()


#The following is used to break out the S/F for a specific number of launches
# look at "devloping_cutset_v0" as well

def cutset_evaluation_spec_veh(target=20, write_csv=False):
    rslt_df = pd.DataFrame()
    plot_df = pd.DataFrame()

    vhcl_fam_list = []
    successes_at_target = []
    failures_at_target = []
    launches_at_target = []
    candidate_vhcl_fam_list = []

    verbose = 0

    Veh_Family=df['Veh_Family'].unique()
    Veh_Family.sort()
    for vhcl_fam in Veh_Family:
        if verbose: print('vhcl=',vhcl_fam)
        rslt_df = df.loc[df['Veh_Family']==vhcl_fam]

        rslt_df.set_index('Date')

        total_successes = max(rslt_df['Success_Bool'].cumsum())
        total_failures = max(rslt_df['Failure_Bool'].cumsum())
        total_launches = total_successes + total_failures

        if total_launches < target:
            if verbose: print(vhcl_fam,'has too few launches')
        else:
            candidate_vhcl_fam_list.append(vhcl_fam)
    if verbose: print('Candidate List at target of',target,':\n',candidate_vhcl_fam_list)

    interim_df = pd.DataFrame()
    rslt_df = pd.DataFrame()
    rslt_df = pd.DataFrame(columns=['vhcl_fam','launches_at_target','successes_at_target','failures_at_target'])

    for vhcl_fam in candidate_vhcl_fam_list:
        if verbose: print('vhcl=',vhcl_fam)
        interim_df = df.loc[df['Veh_Family']==vhcl_fam].copy()
        vhcl_fam_list.append(vhcl_fam)
        interim_df.set_index('Date')
        interim_df['pandas_running_successes'] = interim_df['Success_Bool'].cumsum()
        interim_df['pandas_running_failures'] = interim_df['Failure_Bool'].cumsum()
        interim_df['total_launches']=interim_df['pandas_running_successes'] + interim_df['pandas_running_failures']
        successes_at_target_interim = (interim_df['pandas_running_successes'].loc[interim_df['total_launches']==target].item())
        failures_at_target_interim = (interim_df['pandas_running_failures'].loc[interim_df['total_launches']==target].item())
        launches_at_target_interim = successes_at_target_interim+failures_at_target_interim


        rslt_df = rslt_df.append({'vhcl_fam':vhcl_fam,'launches_at_target':launches_at_target_interim,'successes_at_target':successes_at_target_interim,'failures_at_target':failures_at_target_interim}, ignore_index=True)

    if write_csv == False:
        print(rslt_df[['vhcl_fam','successes_at_target','failures_at_target']].to_string(index=False))


    if write_csv:
        filename='cutset_eval_spec_veh_'+str(target)+'.csv'
        print('Writing to:',filename)
        rslt_df.to_csv(filename, index=False)





cutset_evaluation_spec_veh(target=5, write_csv=True)
cutset_evaluation_spec_veh(target=20, write_csv=True)
cutset_evaluation_spec_veh(target=100, write_csv=True)    


# Steal the cutset for general families from ipynb ("developing_cutset_v0")