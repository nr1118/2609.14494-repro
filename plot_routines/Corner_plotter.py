import neost
from neost.eos import polytropes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner as corner
import seaborn as sns
import matplotlib.patches as mpatches
import os
import pathlib
from pathlib import Path

import argparse

# In[2]:


import neost.global_imports as global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


# In[3]:


import plotting


c_dark_energy = plotting.c_dark_energy
c_bosonic = plotting.c_bosonic
c_fermionic= plotting.c_fermionic
c_baryonic = plotting.c_baryonic[1]


def get_quantiles(array, quantiles=[0.025, 0.5, 0.975]): #0.025,0.5,0.975 0.16,0.5,0.84
        contours = np.nanquantile(array, quantiles) #changed to nanquantile to inorder to ignore the nans that may appear
        low = contours[0]
        median = contours[1]
        high = contours[2]
        minus = low - median
        plus = high - median
        return np.round(median,2),np.round(plus,2),np.round(minus,2) 

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# In[7]:
script_dir = Path(__file__).resolve().parent


plots_path = script_dir.parent / 'plots' if not args.repro else script_dir.parent / 'repro'/ 'plots'
plots_path.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

posterior_data_path = script_dir.parent / 'results'/'posterior'/'DE'/'Dark_energy_posterior_' if not args.repro else script_dir.parent / 'repro'/'Dark_energy_posterior_'/'Dark_energy_posterior_'

prior_data_path = script_dir.parent / 'results'/'prior'/'DE'/'Dark_energy_prior_' if not args.repro else script_dir.parent / 'repro'/'Dark_energy_prior_'/'Dark_energy_prior_'

tmp = np.loadtxt(prior_data_path + 'post_equal_weights.dat')
print('Generating the prior corner plot')


Matrix_prior = np.zeros((len(tmp),3))

#A_param = tmp[:,5]
#rho_plus = tmp[:,6] (units are factors of rho_ns)
#alpha = tmp[:,7]

for i in range(len(tmp)):
    Matrix_prior[i] =tmp[:,5][i], np.log10(tmp[:,6][i]*rho_ns), tmp[:,7][i]

    


figure = corner.corner(Matrix_prior,smooth = 1.0,labels = [r"A",r"log$_{10}$($\varepsilon^+_{dis})$",r"$\alpha$"],
                      range = [(0.1,0.7),(14.4,16),(0.1,1)], show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15},
                      color = '#377eb8',hist_kwargs = {'linestyle': '--','linewidth': 2.0}, contour_kwargs = {'linestyles':'dashed','linewidths': 2.0} )



figure.subplots_adjust(right=1.15,top=1.15)

for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15)


figure.savefig(plots_path / 'Prior_corner_dark_energy.pdf', bbox_inches='tight')


# In[13]:


print('Generating the posterior corner plot')

ewposterior = np.loadtxt(posterior_data_path + 'post_equal_weights.dat')


#A_param = tmp[:,5]
#rho_plus = tmp[:,6] (units are factors of rho_ns)
#alpha = tmp[:,7]

# In[5]:
A = ewposterior[:,5]
rho_plus = ewposterior[:,6]
alpha = ewposterior[:,7]
Matrix = np.zeros((len(ewposterior),3))


for i in range(len(ewposterior)):
    Matrix[i] =A[i],np.log10(rho_plus[i]*rho_ns),alpha[i]


Matrix = np.vstack((Matrix,Matrix,Matrix)) #just stacking the posterior samples three times to make the 1-D histograms be of the same count levels as the priors so that their features are more visible relative to the prior counts. This is just for visualization purposes and does not affect the actual posterior distribution in any way.      
    
ell = corner.corner(Matrix_prior,smooth = 1.0,color = '#377eb8',group = 'prior',range = [(0.1,0.7),(14.4,16),(0.1,1)],
                   plot_datapoints = False,plot_density = True,plot_contours = True,divergences = False,
                    hist_kwargs = {'linestyle': '--','linewidth': 2.0}, contour_kwargs = {'linestyles':'dashed','linewidths': 2.0})


figure = corner.corner(Matrix,smooth = 1.0,fig = ell,color = c_dark_energy,labels = [r"A",r"log$_{10}$($\varepsilon^+_{dis})$",r"$\alpha$"],
                      range = [(0.1,0.7),(14.4,16),(0.1,1)], show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15})

figure.subplots_adjust(right=1.15,top=1.15)

#quantiles =(0.001,0.999)
figure.subplots_adjust(right=1.15,top=1.15)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15) 
    
figure.legend(handles =[matplotlib.lines.Line2D([],[],color = c_dark_energy ,label = 'MCDF Posterior'),
                        matplotlib.lines.Line2D([],[],color = '#377eb8',label = 'MCDF Prior',linestyle = '--',lw = 2.0)],
                  fontsize = 22,frameon = False,loc = "upper right")



figure.savefig(plots_path / 'Posterior_prior_corner_dark_energy.pdf', bbox_inches='tight')



# In[ ]:

bosonic_posterior_data_path = script_dir.parent / 'results'/'posterior'/'BDM'/'Bosonic_posterior_' if not args.repro else script_dir.parent / 'repro'/'Bosonic_posterior_'/'Bosonic_posterior_'

#ADM portion
ewposterior_bosonic = np.loadtxt(bosonic_posterior_data_path + 'post_equal_weights.dat')
print('Generating the prior corner plot')

print(len(ewposterior_bosonic))

# mchi = ewposterior[:,5]
# gchi_over_mphi = ewposterior[:,6]
# Fchi = ewposterior[:,7]
#This is directly related the ordering of variable_params = {'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3],
                #   'mchi':[0, 8],'gchi_over_mphi': [-2,3],'adm_fraction':[0., 5.],'ceft':[EOS.min_norm, EOS.max_norm]}



Matrix_bosonic = np.zeros((len(ewposterior_bosonic),3))

for i in range(len(ewposterior_bosonic)):
    Matrix_bosonic[i] =np.log10(ewposterior_bosonic[:,5][i]),np.log10(ewposterior_bosonic[:,6][i]),ewposterior_bosonic[:,7][i]
    
figure = corner.corner(Matrix_bosonic,smooth = 1.0,labels = [r"log$_{10}$(m$_\chi$/MeV)",r"log$_{10}$($\frac{\mathdefault{g}_\chi}{\mathdefault{m}_\phi/\mathdefault{MeV}})$",r"F$_\chi$ [%]",],
                      range = [(2,8),(-2,3),(0,5.)], show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15},
                      color = '#377eb8',hist_kwargs = {'linestyle': '--','linewidth': 2.0}, contour_kwargs = {'linestyles':'dashed','linewidths': 2.0} )



figure.subplots_adjust(right=1.15,top=1.15)

for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15)




# In[4]:
fermionic_posterior_data_path = script_dir.parent / 'results'/'posterior'/'FDM'/'Fermionic_posterior_' if not args.repro else script_dir.parent / 'repro'/'Fermionic_posterior_'/'Fermionic_posterior_'

ewposterior_fermionic = np.loadtxt(fermionic_posterior_data_path + 'post_equal_weights.dat')


# mchi = ewposterior[:,5]
# gchi_over_mphi = ewposterior[:,6]
# Fchi = ewposterior[:,7]
#This is directly related the ordering of variable_params = {'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3], 'mchi':[0, 9],'gchi_over_mphi': [-5,3],'adm_fraction':[0., 5.],'ceft':[EOS.min_norm, EOS.max_norm]} in the respective run scripts in the run_scripts directory

mchi = ewposterior_fermionic[:,5]
gchi_over_mphi = ewposterior_fermionic[:,6]
Fchi = ewposterior_fermionic[:,7]

Matrix_fermionic = np.zeros((len(ewposterior_fermionic),3))


for i in range(len(ewposterior_fermionic)):
    Matrix_fermionic[i] =np.log10(mchi[i]), np.log10(gchi_over_mphi[i]),Fchi[i]



ell = corner.corner(Matrix_bosonic,smooth = 1.0,color = c_bosonic,range = [(2,8),(-2,3),(0,5.)],
                   plot_datapoints = False,plot_density = True,plot_contours = True,divergences = False,
                    hist_kwargs = {'linestyle': '--','linewidth': 2.0}, contour_kwargs = {'linestyles':'dashed','linewidths': 2.0})


figure = corner.corner(Matrix_fermionic,smooth = 1.0,fig = ell,color = c_fermionic,labels = [r"log$_{10}$(m$_\chi$/MeV)",r"log$_{10}$($\frac{\mathdefault{g}_\chi}{\mathdefault{m}_\phi/\mathrm{MeV}})$",r"F$_\chi \, [\%]$"],
                      range = [(2,8),(-5,3),(0,5.)], show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15},
                      hist_kwargs = {'linestyle': 'dotted','linewidth': 2.0}, contour_kwargs = {'linestyles':'dotted','linewidths': 2.0})
figure.subplots_adjust(right=1.15,top=1.15)


figure.subplots_adjust(right=1.15,top=1.15)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15) 
    
figure.legend(handles =[matplotlib.lines.Line2D([],[],color = c_fermionic,label = 'Fermionic ADM', linestyle = 'dotted', lw=2.0),
                        matplotlib.lines.Line2D([],[],color = c_bosonic,label = 'Bosonic',linestyle = '--',lw = 2.0)],
                  fontsize = 22,frameon = False,loc = "upper right")


figure.savefig('plots/Posterior_corner_bosonic_fermionic_adm.pdf',bbox_inches='tight')


