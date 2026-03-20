import neost
from neost.eos import polytropes
from neost.Prior import Prior
from neost.Star import Star
from neost.Likelihood import Likelihood
from neost import PosteriorAnalysis
from scipy.stats import multivariate_normal, gaussian_kde
import numpy as np
from pymultinest.solve import solve
import time
import os
import pathlib
from pathlib import Path


import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons

script_dir = Path(__file__).resolve().parent

data_path = script_dir.parent / "data"

eos_name = 'polytropes'

EOS = polytropes.PolytropicEoS(crust = 'ceft-Keller-N3LO', rho_t = 1.5*rho_ns, adm_type = 'Fermionic')



# Create the likelihoods for the individual measurements
mr_J0740 = np.loadtxt(f'{data_path}/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat').T
J0740_LL = gaussian_kde(mr_J0740)

mr_J0030 = np.loadtxt(f'{data_path}/J0030_bravo_STPDT_NxX_lp1k_se08_mrsamples_post_equal_weights.dat').T
J0030_LL = gaussian_kde(mr_J0030)


mr_J0437 = np.loadtxt(f'{data_path}/J0437_3C50_CST_PDT_AGN_lp20k_se03_mrsamples_post_equal_weights.dat').T
J0437_LL = gaussian_kde(mr_J0437)


likelihood_functions = [J0740_LL.pdf, J0030_LL.pdf,J0437_LL.pdf]
likelihood_params = [['Mass', 'Radius'],['Mass','Radius'],['Mass','Radius']]

# This is not a GW event so we set chirp mass to None
chirp_mass = [None,None,None]
number_stars = len(chirp_mass)

run_name = "Fermionic_posterior_"
repro_path = script_dir.parent / f'{run_name}/'
repro_path.mkdir(parents=True, exist_ok=True).mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

print(f"Folder created at: {repro_path}")


variable_params = {'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3],
                  'mchi':[0, 9],'gchi_over_mphi': [-5,3],'adm_fraction':[0., 5.],'ceft':[EOS.min_norm, EOS.max_norm]}

for i in range(number_stars):
	variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})




static_params = {}
# In[ ]:


prior = Prior(EOS, variable_params, static_params, chirp_mass)
likelihood = Likelihood(prior, likelihood_functions, likelihood_params, chirp_mass)

print("Bounds of prior are")
print(variable_params)
print("with model "+eos_name)
print("number of parameters is %d" %len(variable_params))

## TESTING ##
print("Testing prior and likelihood")
cube = np.random.rand(50, len(variable_params))
for i in range(len(cube)):
    par = prior.inverse_sample(cube[i])
    print(likelihood.call(par))
print("Testing done")


# In[ ]:


start = time.time()
result = solve(LogLikelihood=likelihood.call, Prior=prior.inverse_sample, n_live_points=3000, evidence_tolerance=0.1,
               n_dims=len(variable_params), sampling_efficiency=0.8, outputfiles_basename=f'{repro_path}/{run_name}', verbose=True)
end = time.time()
print(end - start)

print('Testing done')
print('Moving on to posterior analysis')


PosteriorAnalysis.compute_auxiliary_data(repro_path, EOS, variable_params, static_params, chirp_mass, dm=True, de=False, sampler='multinest', identifier=run_name)

PosteriorAnalysis.compute_table_data(repro_path, EOS, variable_params, static_params, dm=True, de=False, sampler='multinest', identifier=run_name)


def get_quantiles(array, quantiles=[0.025, 0.5, 0.975]):
        contours = np.nanquantile(array, quantiles) #changed to nanquantile to inorder to ignore the nans that may appear
        low = contours[0]
        median = contours[1]
        high = contours[2]
        minus = low - median
        plus = high - median
        return np.round(median,2),np.round(plus,2),np.round(minus,2) 

Data_array = np.loadtxt(repro_path/f'{run_name}' + 'table_data.txt')
print('M_TOV: ', get_quantiles(Data_array[:,0]))
print('R_TOV: ', get_quantiles(Data_array[:,1]))
print('R_1.4: ', get_quantiles(Data_array[:,2]))
print('R_2.0: ', get_quantiles(Data_array[:,3]))
print('Delta R = R_2.0 - R_1.4: ', get_quantiles(Data_array[:,3] - Data_array[:,2]))
