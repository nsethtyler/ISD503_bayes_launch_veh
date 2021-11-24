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
import sys
from matplotlib.ticker import NullFormatter, FixedLocator


try:
	if sys.argv[1] == '-company':
		company_of_interest = sys.argv[2]
except:
	company_of_interest = 'SpaceX'
	pass

# Setup environment
# plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12,8)

save_figs = 0
show_all_companies = 0



datafile = '../08-Datasets/Aggregated_SLR_Dataset_v4.xlsx'



# Old model, "self built"
# def beta_dist(alpha, beta):
# 	x = np.linspace(0, 1, num=100)
# 	cap_Beta = (gamma(alpha+beta) / (gamma(alpha)*gamma(beta)))
# 	dist = (x**(alpha-1))*((1-x)**(beta-1)) / cap_Beta
# 	return dist


def calc_betas(successes, failures):
	'''
	Returns the PDF and relevant stats given number of past and new successes
	Based upon:
	kdnuggets.com/2019/09/beta-distribution-what-when-how.html
	'''
	loc = 0
	scale = 1
	x = np.linspace(0,1,1000)
	lower_bound_limit = 0.05 #decimal
	upper_bound_limit = 0.95 #decimal

	alpha = failures
	beta = successes

	# y = ss.beta.pdf(x, alpha, beta, loc, scale)

	
	lower_cred_int = btdtri(alpha, beta, lower_bound_limit)
	upper_cred_int = btdtri(alpha, beta, upper_bound_limit)

	return lower_cred_int, upper_cred_int

#Function for plotting
def forward(x):
	return x**(1/2)

def inverse(x):
	return x**2


sheet = 'All'

# Import data for evaluation
df = pd.read_excel(datafile, sheet_name=sheet,engine="openpyxl")
df.set_index('Date')

df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])


# Summary - Detailed

rslt_df=df.copy()
rslt_df.set_index('Date')

rslt_df['pandas_success']=rslt_df['Success_Bool'].cumsum(axis=0)
rslt_df['pandas_failure']=rslt_df['Failure_Bool'].cumsum(axis=0)
rslt_df['pandas_MLE']=rslt_df['pandas_failure']/(rslt_df['pandas_failure']+rslt_df['pandas_success'])

for index,row in rslt_df.iterrows():
    # row['lower_cred_int'], row['upper_cred_int'] = calc_betas(int(row['pandas_success']),int(row['pandas_failure']))
    # print(row['pandas_success'], row['pandas_failure'])
    success_count = row['pandas_success']
    failure_count = row['pandas_failure']
    lower_cred_int, upper_cred_int = calc_betas(success_count,failure_count)
    rslt_df.loc[index,'lower_cred_int']=lower_cred_int
    rslt_df.loc[index,'upper_cred_int']=upper_cred_int
    # print(lower_cred_int, row['lower_cred_int'])

rslt_df['lower_cred_int'].fillna(0)
rslt_df['upper_cred_int'].fillna(1)
rslt_df.set_index('Date')


fig, axs = plt.subplots(2,1)
axs[0].plot(rslt_df['Date'],rslt_df['pandas_success']+rslt_df['pandas_failure'])
axs[0].grid(True)
axs[0].set_ylabel('Launch Attempts')
for index,row in rslt_df.iterrows():
    if row['Failure_Bool']:
        axs[0].scatter(row['Date'],row['pandas_success']+row['pandas_failure'], 120, color='red', marker='+')

axs[1].plot(rslt_df['Date'],rslt_df['pandas_MLE'], label='MLE')
axs[1].plot(rslt_df['Date'],rslt_df['lower_cred_int'], linestyle='--', color='red')
axs[1].plot(rslt_df['Date'],rslt_df['upper_cred_int'], linestyle='--', color='red')
axs[1].fill_between(rslt_df['Date'],rslt_df['lower_cred_int'],rslt_df['upper_cred_int'], color='mistyrose', label='Credible Interval')
axs[1].grid(True)
axs[1].set_ylabel('Probability of Failure')
axs[1].set_xlabel('Date')
axs[1].set_yscale('function',functions=(forward, inverse))
axs[1].yaxis.set_major_locator(FixedLocator(np.arange(0, 1.0, 0.1)**2))
axs[1].yaxis.set_major_locator(FixedLocator(np.arange(0, 1.0, 0.1)))


for i in range(0,2):
    axs[i].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

for i in range(0,1):
    plt.setp(axs[i].get_xticklabels(), visible=False)

plt.subplots_adjust(wspace=0.25,hspace=.1)


plot_summary = 'Total Successes: ' + str(rslt_df['pandas_success'].max()) + '\nTotal Failures: ' + str(rslt_df['pandas_failure'].max())
plt.figtext(0.8,0.92,plot_summary,style='italic', color='slategray')


plt.suptitle('Overall - 1998-2020')
plt.legend()

if save_figs:
	savefilename='03-Figures/Overall.png'
	plt.savefig(savefilename)
else:
	plt.show()










# By Manufacturer
if show_all_companies:


	Manufacturers=df['Veh_Mfctr'].unique()
	Manufacturers.sort()
	company_number = 0
	for company_of_interest in Manufacturers:
		rslt_df=df.loc[df['Veh_Mfctr'] == company_of_interest].copy()
		rslt_df.set_index('Date')

		rslt_df['pandas_success']=rslt_df['Success_Bool'].cumsum(axis=0)
		rslt_df['pandas_failure']=rslt_df['Failure_Bool'].cumsum(axis=0)
		rslt_df['pandas_MLE']=rslt_df['pandas_failure']/(rslt_df['pandas_failure']+rslt_df['pandas_success'])

		for index,row in rslt_df.iterrows():
			# row['lower_cred_int'], row['upper_cred_int'] = calc_betas(int(row['pandas_success']),int(row['pandas_failure']))
			# print(row['pandas_success'], row['pandas_failure'])
			success_count = row['pandas_success']
			failure_count = row['pandas_failure']
			lower_cred_int, upper_cred_int = calc_betas(success_count,failure_count)
			rslt_df.loc[index,'lower_cred_int']=lower_cred_int
			rslt_df.loc[index,'upper_cred_int']=upper_cred_int
			# print(lower_cred_int, row['lower_cred_int'])

		rslt_df['lower_cred_int'].fillna(0)
		rslt_df['upper_cred_int'].fillna(1)
		rslt_df.set_index('Date')


		fig, axs = plt.subplots(2,1)
		axs[0].plot(rslt_df['Date'],rslt_df['pandas_success']+rslt_df['pandas_failure'])
		axs[0].grid(True)
		axs[0].set_ylabel('Launch Attempts')
		for index,row in rslt_df.iterrows():
			if row['Failure_Bool']:
				axs[0].scatter(row['Date'],row['pandas_success']+row['pandas_failure'], 120, color='red', marker='+')

		axs[1].plot(rslt_df['Date'],rslt_df['pandas_MLE'], label='MLE')
		axs[1].plot(rslt_df['Date'],rslt_df['lower_cred_int'], linestyle='--', color='red')
		axs[1].plot(rslt_df['Date'],rslt_df['upper_cred_int'], linestyle='--', color='red')
		axs[1].fill_between(rslt_df['Date'],rslt_df['lower_cred_int'],rslt_df['upper_cred_int'], color='mistyrose', label='Credible Interval')
		axs[1].grid(True)
		axs[1].set_ylabel('Probability of Failure')


		for i in range(0,2):
			axs[i].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

		for i in range(0,1):
			plt.setp(axs[i].get_xticklabels(), visible=False)

		plt.subplots_adjust(wspace=0.25,hspace=.1)


		plot_summary = 'Total Successes: ' + str(rslt_df['pandas_success'].max()) + '\nTotal Failures: ' + str(rslt_df['pandas_failure'].max())
		plt.figtext(0.8,0.92,plot_summary,style='italic', color='slategray')


		plt.suptitle(company_of_interest)
		plt.legend()
		if save_figs:
			company_number += 1
			savefilename='03-Figures/'+str(company_number) + '.png'
			plt.savefig(savefilename)
			print(company_of_interest, 'is plot number', company_number)
		else:
			plt.show()




# Summary - All Time


df = pd.DataFrame()
sheet = 'Total_Annual_Launches'

# Import data for evaluation
df = pd.read_excel(datafile, sheet_name=sheet,engine="openpyxl")
df.set_index('Date')

df = df.reset_index()
# df['Date'] = pd.to_datetime(df['Date'])


rslt_df=df.copy()
rslt_df.set_index('Date')

rslt_df['pandas_success']=rslt_df['Successes'].cumsum(axis=0)
rslt_df['pandas_failure']=rslt_df['Failures'].cumsum(axis=0)
rslt_df['pandas_MLE']=rslt_df['pandas_failure']/(rslt_df['pandas_failure']+rslt_df['pandas_success'])

for index,row in rslt_df.iterrows():
    # row['lower_cred_int'], row['upper_cred_int'] = calc_betas(int(row['pandas_success']),int(row['pandas_failure']))
    # print(row['pandas_success'], row['pandas_failure'])
    success_count = row['pandas_success']
    failure_count = row['pandas_failure']
    lower_cred_int, upper_cred_int = calc_betas(success_count,failure_count)
    rslt_df.loc[index,'lower_cred_int']=lower_cred_int
    rslt_df.loc[index,'upper_cred_int']=upper_cred_int
    # print(lower_cred_int, row['lower_cred_int'])

rslt_df['lower_cred_int'].fillna(0)
rslt_df['upper_cred_int'].fillna(1)
rslt_df.set_index('Date')


fig, axs = plt.subplots(2,1)
axs[0].plot(rslt_df['Date'],rslt_df['pandas_success']+rslt_df['pandas_failure'])
axs[0].grid(True)
axs[0].set_ylabel('Launch Attempts')
# for index,row in rslt_df.iterrows():
#     if row['Failure_Bool']:
#         axs[0].scatter(row['Date'],row['pandas_success']+row['pandas_failure'], 120, color='red', marker='+')

axs[1].plot(rslt_df['Date'],rslt_df['pandas_MLE'], label='MLE')
axs[1].plot(rslt_df['Date'],rslt_df['lower_cred_int'], linestyle='--', color='red')
axs[1].plot(rslt_df['Date'],rslt_df['upper_cred_int'], linestyle='--', color='red')
axs[1].fill_between(rslt_df['Date'],rslt_df['lower_cred_int'],rslt_df['upper_cred_int'], color='mistyrose', label='Credible Interval')
axs[1].grid(True)
axs[1].set_ylabel('Probability of Failure')
axs[1].set_yscale('function',functions=(forward, inverse))
axs[1].yaxis.set_major_locator(FixedLocator(np.arange(0, 1.0, 0.1)**2))
axs[1].yaxis.set_major_locator(FixedLocator(np.arange(0, 1.0, 0.1)))

for i in range(0,2):
    axs[i].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

for i in range(0,1):
    plt.setp(axs[i].get_xticklabels(), visible=False)

plt.subplots_adjust(wspace=0.25,hspace=.1)


plot_summary = 'Total Successes: ' + str(rslt_df['pandas_success'].max()) + '\nTotal Failures: ' + str(rslt_df['pandas_failure'].max())
plt.figtext(0.8,0.92,plot_summary,style='italic', color='slategray')


plt.suptitle('Overall - 1957-2020')
plt.legend()

if save_figs:
	savefilename='03-Figures/Overall-All_Time.png'
	plt.savefig(savefilename)
else:
	plt.show()