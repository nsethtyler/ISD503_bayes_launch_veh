'''
Seth Tyler
ISD503 Project
Fall 2021
'''

'''
bayes_prediction_model_v4.py
This script is used to plot the beta distribution for a given number of successes and failures
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
from scipy.special import gamma, btdtri
import scipy.stats as ss


# Setup environment
plt.rcParams['figure.figsize'] = (10,8)
michigan_blue = '#00274C'
powder_blue = '#E7F3F6'


verbose = 0


def fix_axis():
	plt.ylabel(r'$\it{p(}$$\theta$)')
	plt.xlabel(r'$\theta$')


def calc_prior(prior_successes, prior_failures, new_successes, new_failures):
	'''
	Returns the PDF and relevant stats given passed values of past and new failures
	'''
	loc = 0
	scale = 1
	x = np.linspace(0,1,1000)
	lower_bound_limit = 0.05 #decimal
	upper_bound_limit = 0.95 #decimal

	prior_alpha = prior_failures
	prior_beta = prior_successes

	posterior_alpha = prior_failures + new_failures
	posterior_beta = prior_successes + new_successes

	if verbose:
		print('Prior alpha / beta:', prior_alpha, '/', prior_beta)
		print('Posterior alpha / beta:', posterior_alpha, '/', posterior_beta)

	prior_y = ss.beta.pdf(x, prior_alpha, prior_beta, loc, scale)
	posterior_y = ss.beta.pdf(x,posterior_alpha, posterior_beta,loc,scale)

	max_prior_y = np.argmax(prior_y)
	max_posterior_y = np.argmax(posterior_y)
	bayes_map_prior_x = x[max_prior_y]
	bayes_map_posterior_x = x[max_posterior_y]


	# prior_mean, prior_var, prior_skew, prior_kurt = ss.beta.stats(prior_alpha,prior_beta, moments = 'mvsk')
	# posterior_mean, posterior_var, posterior_skew, posterior_kurt = ss.beta.stats(posterior_alpha,posterior_beta, moments = 'mvsk')
	
	try:
		prior_MLE = (prior_alpha) / (prior_alpha + prior_beta)
	except:
		prior_MLE = 0
	
	try:
		posterior_MLE = (posterior_alpha) / (posterior_alpha + posterior_beta)
	except:
		posterior_MLE = 0

	if verbose:
		print('Prior / Posterior MLE is ', round(prior_MLE,3), '/', round(posterior_MLE, 3))


	prior_lower_cred_int = btdtri(prior_alpha, prior_beta, lower_bound_limit)
	prior_upper_cred_int = btdtri(prior_alpha, prior_beta, upper_bound_limit)
	posterior_lower_cred_int = btdtri(posterior_alpha, posterior_beta, lower_bound_limit)
	posterior_upper_cred_int = btdtri(posterior_alpha, posterior_beta, upper_bound_limit)


	return x, prior_y, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int


def plot_betas(x, prior_y, posterior_y, posterior_MLE, bayes_map_posterior_x, posterior_lower_cred_int, posterior_upper_cred_int, title):
	'''
	Plots the prior beta distribution and posterior beta distribution, along with the MLE and maximum bayesian estimate
	'''
	prior_label = r'prior: $\alpha_{prior}$ = ' + str(prior_failures) + r', $\beta_{prior}$ = ' + str(prior_successes)
	posterior_label = r'posterior: $\alpha_{posterior}$ = ' + str(prior_failures + new_failures) + r', $\beta_{posterior}$ = ' + str(prior_successes + new_successes)
	posterior_MLE_info='posterior MLE = '+str(round(posterior_MLE,3))
	bayes_map_posterior_x_info='Bayes MAP Estimate = '+str(round(bayes_map_posterior_x,3))
	plt.plot(x, prior_y, label=prior_label, color='lightblue', linestyle='--')
	plt.plot(x, posterior_y, label=posterior_label, color='darkblue')
	plt.axvline(posterior_MLE, color='red',linestyle='--', label=posterior_MLE_info)
	plt.axvline(bayes_map_posterior_x, color='orange',linestyle='--', label=bayes_map_posterior_x_info)

	legend = plt.legend(loc='best', shadow=True, fontsize='large')
	if len(title) > 3:
		title_string = str(title)
	else:
		title_string = 'Bayesian Updating\nNew Successes / Failures: ' + str(new_successes) + ' / ' + str(new_failures)
	plt.fill_between(x,posterior_y,0,where=(x>posterior_lower_cred_int) & (x<=posterior_upper_cred_int), color='mistyrose')
	plt.title(title_string)
	plt.grid()
	fix_axis()
	plt.show()


# #File name & Location (relative to this file)

datafile = '../08-Datasets/Aggregated_SLR_Dataset_v4.xlsx'
datatab = 'Launch_Shares_Mftr'

# Import data for evaluation
df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")

df = df.reset_index()



plots_or_not = 0
if plots_or_not:
	prior_successes = 0
	prior_failures = 0
	new_successes = 5560
	new_failures = 494
	company = "Total - Random"

	x, prior_y, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)

	plot_betas(x, prior_y, posterior_y, posterior_MLE, bayes_map_posterior_x, posterior_lower_cred_int, posterior_upper_cred_int, title=company)




# To plot example for document

plot_dist_example = 0
if plot_dist_example:
	prior_successes = 2
	prior_failures = 6
	new_successes = 3
	new_failures = 0

	company = "Prior / Posterior Beta Distribution Example"

	x, prior_y, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)

	plot_betas(x, prior_y, posterior_y, posterior_MLE,bayes_map_posterior_x, posterior_lower_cred_int, posterior_upper_cred_int, title=company)




# Plot various betas for comparison 

prior_info_impact_comparison = 1
if prior_info_impact_comparison:
	prior_successes = 1
	prior_failures = 1
	new_successes = 0
	new_failures = 0


	x, prior_y_uni, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)

	prior_successes = 8
	prior_failures = 2

	x, prior_y_2_8, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)

	prior_successes = 28
	prior_failures = 2

	x, prior_y_2_28, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)


	## Plotting
	plt.plot(x, prior_y_uni, label=r'Uniform Prior: $\alpha$ = $\beta$ = 1', color=michigan_blue)
	plt.plot(x, prior_y_2_8, label=r'Lightly Informed Prior: $\alpha$ = 2,  $\beta$ = 8', color=michigan_blue, linestyle='-.')
	plt.plot(x, prior_y_2_28, label=r'Heavily Informed Prior: $\alpha$ = 2,  $\beta$ = 28', color=michigan_blue, linestyle=':')

	legend = plt.legend(loc='best', shadow=True, fontsize='large')
	plt.title('Comparison of Various Beta Distributions')
	plt.grid()
	fix_axis()

	plt.tight_layout()

	plt.show()



# Plot various betas for sensitivity - uniform/uninformed to small sample

prior_info_impact_sensitivity_uni_to_small = 1
if prior_info_impact_sensitivity_uni_to_small:
	prior_successes = 1
	prior_failures = 1
	new_successes = 0
	new_failures = 2


	x, prior_y_uni, posterior_y_uni, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)

	
	## Plotting
	plt.plot(x, prior_y_uni, label=r'Uniform Prior: $\alpha$ = $\beta$ = 1', color=michigan_blue, linestyle=':')
	plt.plot(x, posterior_y_uni, label=r'Posterior w/ 2 new failures', color=michigan_blue, linestyle='-')

	# bayes_map_prior_x_info='Bayes max prior = '+str(round(bayes_map_prior_x,3))
	# bayes_map_posterior_x_info='Bayes max posterior = '+str(round(bayes_map_posterior_x,3))
	# plt.axvline(bayes_map_prior_x, color='red',linestyle='--', label=bayes_map_prior_x_info)
	# plt.axvline(bayes_map_posterior_x, color='orange',linestyle='--', label=bayes_map_posterior_x_info)

	legend = plt.legend(loc='best', shadow=True, fontsize='medium')
	plt.title('Uninformed Prior Impact on Posterior')
	plt.grid(alpha=0.5)
	fix_axis()

	plt.tight_layout()

	plt.fill_between(x,posterior_y_uni,0,where=(x>posterior_lower_cred_int) & (x<=posterior_upper_cred_int), color=powder_blue)

	plt.show()
	print("uni",posterior_lower_cred_int, posterior_upper_cred_int)



# Plot various betas for sensitivity - lightly informed to small sample

prior_info_impact_sensitivity_light_to_small = 1
if prior_info_impact_sensitivity_light_to_small:

	prior_successes = 8
	prior_failures = 2
	new_successes = 0
	new_failures = 1

	x, prior_y_2_8, posterior_y_2_8, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)
	bayes_max_delta = bayes_map_posterior_x - bayes_map_prior_x

	
	## Plotting
	plt.plot(x, prior_y_2_8, label=r'Lightly Informed Prior: $\alpha$ = 2, $\beta$ = 8', color=michigan_blue, linestyle=':')
	plt.plot(x, posterior_y_2_8, label=r'Posterior with 1 new failure', color=michigan_blue, linestyle='-')
	
	bayes_map_prior_x_info='Prior MAP Estimate = '+str(round(bayes_map_prior_x,3))
	bayes_map_posterior_x_info='Posterior MAP Estimate = '+str(round(bayes_map_posterior_x,3))
	plt.axvline(bayes_map_prior_x, color='red',linestyle='--', label=bayes_map_prior_x_info)
	plt.axvline(bayes_map_posterior_x, color='orange',linestyle='--', label=bayes_map_posterior_x_info)

	xmin,xmax,ymin,ymax = plt.axis()
	prob_shift_text = 'Max Probability Shift = ' + str(round(bayes_max_delta,3))
	props = dict(boxstyle='round', facecolor='white', alpha=0.5)
	plt.text(1.15 *  max(bayes_map_posterior_x, bayes_map_prior_x),0.9*ymax ,prob_shift_text, fontstyle='italic', fontweight='bold', bbox=props)

	plt.fill_between(x,posterior_y_2_8,0,where=(x>posterior_lower_cred_int) & (x<=posterior_upper_cred_int), color=powder_blue)
	
	legend = plt.legend(loc='best', shadow=True, fontsize='medium')
	plt.title('Lightly Informed Prior Impact on Posterior')
	fix_axis()

	plt.grid(alpha=0.5)
	plt.tight_layout()

	plt.show()

	print("light",posterior_lower_cred_int, posterior_upper_cred_int)



# Plot various betas for sensitivity - heavily informed to small sample

prior_info_impact_sensitivity_heavy_to_small = 1
if prior_info_impact_sensitivity_heavy_to_small:

	prior_successes = 28
	prior_failures = 2
	new_successes = 0
	new_failures = 1

	x, prior_y_2_28, posterior_y_2_28, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)
	bayes_max_delta = bayes_map_posterior_x - bayes_map_prior_x
	
	## Plotting
	plt.plot(x, prior_y_2_28, label=r'Heavily Informed Prior: $\alpha$ = 2, $\beta$ = 28', color=michigan_blue, linestyle=':')
	plt.plot(x, posterior_y_2_28, label=r'Posterior with 1 new failure', color=michigan_blue, linestyle='-')
	
	bayes_map_prior_x_info='Prior MAP Estimate = '+str(round(bayes_map_prior_x,3))
	bayes_map_posterior_x_info='Posterior MAP Estimate = '+str(round(bayes_map_posterior_x,3))
	plt.axvline(bayes_map_prior_x, color='red',linestyle='--', label=bayes_map_prior_x_info)
	plt.axvline(bayes_map_posterior_x, color='orange',linestyle='--', label=bayes_map_posterior_x_info)

	xmin,xmax,ymin,ymax = plt.axis()
	prob_shift_text = 'Max Probability Shift = ' + str(round(bayes_max_delta,3))
	props = dict(boxstyle='round', facecolor='white', alpha=0.5)
	plt.text(1.2 * max(bayes_map_posterior_x, bayes_map_prior_x),0.9*ymax ,prob_shift_text, fontstyle='italic', fontweight='bold', bbox=props)

	plt.fill_between(x,posterior_y_2_28,0,where=(x>posterior_lower_cred_int) & (x<=posterior_upper_cred_int), color=powder_blue)

	legend = plt.legend(loc='best', shadow=True, fontsize='medium')
	plt.title('Heavily Informed Prior Impact on Posterior')
	# plt.xlim(0,0.4)
	fix_axis()

	plt.grid(alpha=0.5)
	plt.tight_layout()

	plt.show()

	print("heavy",posterior_lower_cred_int, posterior_upper_cred_int)


# Plot First 20 launches of any vehicle
# #File name & Location (relative to this file)

datafile = '../08-Datasets/Aggregated_SLR_Dataset_v4.xlsx'
datatab = 'All'

# Import data for evaluation
df = pd.read_excel(datafile, sheet_name=datatab,engine="openpyxl")

df = df.reset_index()





# Plot sample beta distribution

plot_sample_depiction = 1
if plot_sample_depiction:

	prior_successes = 8
	prior_failures = 2
	new_successes = 0
	new_failures = 4

	x, prior_y, posterior_y, prior_MLE, posterior_MLE, bayes_map_prior_x, bayes_map_posterior_x, prior_lower_cred_int, prior_upper_cred_int, posterior_lower_cred_int, posterior_upper_cred_int = calc_prior(prior_successes, prior_failures, new_successes, new_failures)
	bayes_max_delta = bayes_map_posterior_x - bayes_map_prior_x
	
	## Plotting
	plt.plot(x, prior_y, label=r'Prior', color=michigan_blue, linestyle=':')
	plt.plot(x, posterior_y, label=r'Posterior', color=michigan_blue, linestyle='-')
	
	bayes_map_prior_x_info='Prior MAP Estimate = '+str(round(bayes_map_prior_x,3))
	bayes_map_posterior_x_info='Posterior MAP Estimate = '+str(round(bayes_map_posterior_x,3))
	# plt.axvline(bayes_map_prior_x, color='red',linestyle='--', label=bayes_map_prior_x_info)
	# plt.axvline(bayes_map_posterior_x, color='orange',linestyle='--', label=bayes_map_posterior_x_info)
	plt.annotate('Posterior MAP', xy=(bayes_map_posterior_x,max(posterior_y)), xytext=(bayes_map_posterior_x+0.1,max(posterior_y)+0.5), arrowprops=dict(facecolor=michigan_blue, shrink=0.05),)
	plt.fill_between(x,posterior_y,0,where=(x>posterior_lower_cred_int) & (x<=posterior_upper_cred_int), color=powder_blue)
	plt.annotate('Credible Interval (HPDI)', xy=(posterior_upper_cred_int-0.1,max(posterior_y)/2), xytext=(posterior_upper_cred_int+0.1,max(posterior_y)-0.5), arrowprops=dict(facecolor=michigan_blue, shrink=0.05),)

	plt.plot(bayes_map_posterior_x,max(posterior_y),'|', color=michigan_blue)

	xmin,xmax,ymin,ymax = plt.axis()
	# prob_shift_text = 'Max Probability Shift = ' + str(round(bayes_max_delta,3))
	# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
	# plt.text(1.2 * max(bayes_map_posterior_x, bayes_map_prior_x),0.9*ymax ,prob_shift_text, fontstyle='italic', fontweight='bold', bbox=props)

	legend = plt.legend(loc='best', shadow=True, fontsize='medium')
	plt.title('General Example of Beta Distribution Nomenclature')
	# plt.xlim(0,0.4)
	fix_axis()

	plt.grid(alpha=0.5)
	plt.tight_layout()

	plt.show()



