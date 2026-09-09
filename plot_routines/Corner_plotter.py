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

#This is directly related the ordering of variable_params = {'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3],'A_param':[0.1, 0.7],'rho_plus': [1.5,37.31426766],'alpha':[0.1, 1.],'ceft':[EOS.min_norm, EOS.max_norm]} in the respective run scripts in the run_scripts directory.


# In[5]:
A = ewposterior[:,5]
rho_plus = ewposterior[:,6]
alpha = ewposterior[:,7]
Matrix = np.zeros((len(ewposterior),3))


for i in range(len(ewposterior)):
    Matrix[i] =A[i],np.log10(rho_plus[i]*rho_ns),alpha[i]


Matrix = np.vstack((Matrix,Matrix)) #just stacking the posterior samples two times to make the 1-D histograms be of the same count levels as the priors so that their features are more visible relative to the prior counts. This is just for visualization purposes and does not affect the actual posterior distribution in any way.      
    
ell = corner.corner(Matrix_prior,smooth = 1.0,color = '#377eb8',group = 'prior',range = [(0.1,0.7),(14.4,16),(0.1,1)],
                   plot_datapoints = False,plot_density = True,plot_contours = True,divergences = False,levels=(0.393, 0.864),
                    hist_kwargs = {'linestyle': '--','linewidth': 2.0}, contour_kwargs = {'linestyles':'dashed','linewidths': 2.0})


figure = corner.corner(Matrix,smooth = 1.0,fig = ell,color = c_dark_energy,labels = [r"A",r"log$_{10}$($\varepsilon^+_{dis})$",r"$\alpha$"],
                      range = [(0.1,0.7),(14.4,16),(0.1,1)], levels=(0.393, 0.864), show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15})

#Here, the levels argument specifies the contour levels for the posterior/prior distribution. The values 0.393 and 0.864 correspond to the 1-sigma and 2-sigma confidence intervals, respectively, for a 2-D Gaussian distribution. You can adjust these values based on your specific requirements or desired confidence levels. For example, if you want the 68% and 95% confidence intervals, you can use levels=(0.68, 0.95) instead. The choice of contour levels depends on the choice of the user. 

figure.subplots_adjust(right=1.15,top=1.15)


figure.subplots_adjust(right=1.15,top=1.15)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15) 
    
figure.legend(handles =[matplotlib.lines.Line2D([],[],color = c_dark_energy ,label = 'MCDF Posterior'),
                        matplotlib.lines.Line2D([],[],color = '#377eb8',label = 'MCDF Prior',linestyle = '--',lw = 2.0)],
                  fontsize = 22,frameon = False,loc = "upper right")



figure.savefig(plots_path / 'Posterior_prior_corner_dark_energy.pdf', bbox_inches='tight')



#Bosonic ADM portion

bosonic_prior_data_path = script_dir.parent / 'results'/'prior'/'BDM'/'Bosonic_prior_' if not args.repro else script_dir.parent / 'repro'/'Bosonic_prior_'/'Bosonic_prior_'

ewprior_bosonic = np.loadtxt(bosonic_prior_data_path + 'post_equal_weights.dat')



Matrix_bosonic_prior = np.zeros((len(ewprior_bosonic),3))

for i in range(len(ewprior_bosonic)):
    Matrix_bosonic_prior[i] =np.log10(ewprior_bosonic[:,5][i]),np.log10(ewprior_bosonic[:,6][i]),ewprior_bosonic[:,7][i]

bosonic_posterior_data_path = script_dir.parent / 'results'/'posterior'/'BDM'/'Bosonic_posterior_' if not args.repro else script_dir.parent / 'repro'/'Bosonic_posterior_'/'Bosonic_posterior_'

#ADM portion
ewposterior_bosonic = np.loadtxt(bosonic_posterior_data_path + 'post_equal_weights.dat')



# mchi = ewposterior[:,5]
# gchi_over_mphi = ewposterior[:,6]
# Fchi = ewposterior[:,7]
#This is directly related the ordering of variable_params = {'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3],'mchi':[0, 8],'gchi_over_mphi': [-2,3],'adm_fraction':[0., 5.],'ceft':[EOS.min_norm, EOS.max_norm]} in the respective run scripts in the run_scripts directory.



Matrix_bosonic = np.zeros((len(ewposterior_bosonic),3))

for i in range(len(ewposterior_bosonic)):
    Matrix_bosonic[i] =np.log10(ewposterior_bosonic[:,5][i]),np.log10(ewposterior_bosonic[:,6][i]),ewposterior_bosonic[:,7][i]



Matrix_bosonic = np.vstack((Matrix_bosonic,Matrix_bosonic)) #just stacking the posterior samples two times to make the 1-D histograms be of the same count levels as the priors so that their features are more visible relative to the prior counts. This is just for visualization purposes and does not affect the actual posterior distribution in any way.

print('Generating the joint prior and posterior corner plot')


ell = corner.corner(Matrix_bosonic_prior,smooth = 1.0,color = 'xkcd:grey',group = 'prior',range = [(2,8),(-2,3),(0,5.)],
                   plot_datapoints = False,plot_density = True,plot_contours = True,divergences = False, levels=(0.393, 0.864),
                    hist_kwargs = {'linestyle': '-','linewidth': 2.0}, contour_kwargs = {'linestyles':'solid','linewidths': 2.0})


figure = corner.corner(Matrix_bosonic,smooth = 1.0,labels = [r"log$_{10}$(m$_\chi$/MeV)",r"log$_{10}$($\frac{\mathdefault{g}_\chi}{\mathdefault{m}_\phi/\mathdefault{MeV}})$",r"F$_\chi$ [%]",],
                      range = [(2,8),(-2,3),(0,5.)], show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15},
                      color = c_bosonic,hist_kwargs = {'linestyle': '--','linewidth': 2.0}, contour_kwargs = {'linestyles':'dashed','linewidths': 2.0} )



figure.subplots_adjust(right=1.15,top=1.15)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15) 
    
figure.legend(handles =[matplotlib.lines.Line2D([],[],color = 'xkcd:grey',label = 'Bosonic ADM Prior', linestyle = '-', lw=2.0),
                        matplotlib.lines.Line2D([],[],color = c_bosonic,label = 'Bosonic ADM Posterior',linestyle = '--',lw = 2.0)],
                  fontsize = 22,frameon = False,loc = "upper right")


figure.savefig(plots_path / 'Posterior_prior_corner_bosonic_adm.pdf',bbox_inches='tight')


fig,ax = plt.subplots(figsize = (10,11))
my_fontsize=20
my_font = 'serif'
matplotlib.rc('font',family = my_font)
matplotlib.rcParams.update({'font.size': 20})
bosonic_ratio = np.log10(10**Matrix[:,1]/10**Matrix[:,0])

print('The bosonic ratio of gchi/mphi to mchi is given by log10(gchi/mphi/mchi), which is given in the main text of the paper. The 68% and 95% confidence intervals for this ratio are given below.')

print('Posteriors 68% ', get_quantiles(bosonic_ratio, quantiles = [0.16,0.5,0.84]) )
print('Posteriors 95% ', get_quantiles(bosonic_ratio))
plot = sns.kdeplot(x = bosonic_ratio,y = Matrix[:,2],shade = True,cbar = False,cmap = 'Purples_r'
                   ,common_norm = True,levels=[0.05,0.32,1.],ax = ax)


bosonic_ratio_prior = np.log10(10**Matrix_prior[:,1]/10**Matrix_prior[:,0])
plot_prior = sns.kdeplot(x = bosonic_ratio_prior,y = Matrix_prior[:,2],shade = False,cbar = False,
                   colors = 'xkcd:grey',common_norm = True,levels =[0.05,0.32,1],label = "Prior", linestyles = 'solid',
                        linewidths = 2.,ax = ax)




ax.legend(handles =[mpatches.Patch(color = c_bosonic ,label = 'Bosonic Posterior')],
                  fontsize = 18,frameon = True,loc = "upper left")


ax.set_ylim(0.0,5.5)
ax.set_xlim(-8,-3)
ax.set_yticks(np.arange(0, 5.5,.5),minor =True)
ax.set_xticks([-8,-7,-6,-5,-4,-3], minor = True)
ax.set_ylabel(r"F$_\chi$ [$\%$]",fontsize=my_fontsize,font = my_font)
ax.set_xlabel(r"$\log_{10}(\frac{\mathdefault{g}_\chi}{\mathdefault{m}_\phi/\mathrm{MeV}}/(\mathdefault{m}_\chi/\mathrm{MeV}))$",fontsize=my_fontsize,font = my_font)
ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)



fig.savefig(plots_path / 'bosonic_adm_ratio_plot.pdf',bbox_inches='tight')


#Fermionic ADM portion

fermionic_prior_data_path = script_dir.parent / 'results'/'prior'/'FDM'/'Fermionic_prior_' if not args.repro else script_dir.parent / 'repro'/'Fermionic_prior_'/'Fermionic_prior_'

ewprior_fermionic = np.loadtxt(fermionic_prior_data_path + 'post_equal_weights.dat')



Matrix_fermionic_prior = np.zeros((len(ewprior_fermionic),3))

for i in range(len(ewprior_fermionic)):
    Matrix_fermionic_prior[i] =np.log10(ewprior_fermionic[:,5][i]),np.log10(ewprior_fermionic[:,6][i]),ewprior_fermionic[:,7][i]


fermionic_posterior_data_path = script_dir.parent / 'results'/'posterior'/'FDM'/'Fermionic_posterior_' if not args.repro else script_dir.parent / 'repro'/'Fermionic_posterior_'/'Fermionic_posterior_'

ewposterior_fermionic = np.loadtxt(fermionic_posterior_data_path + 'post_equal_weights.dat')


# mchi = ewposterior[:,5]
# gchi_over_mphi = ewposterior[:,6]
# Fchi = ewposterior[:,7]
#This is directly related the ordering of variable_params = {'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2':[2.,8.3], 'mchi':[0, 9],'gchi_over_mphi': [-5,3],'adm_fraction':[0., 5.],'ceft':[EOS.min_norm, EOS.max_norm]} in the respective run scripts in the run_scripts directory.

mchi = ewposterior_fermionic[:,5]
gchi_over_mphi = ewposterior_fermionic[:,6]
Fchi = ewposterior_fermionic[:,7]

Matrix_fermionic = np.zeros((len(ewposterior_fermionic),3))


for i in range(len(ewposterior_fermionic)):
    Matrix_fermionic[i] =np.log10(mchi[i]), np.log10(gchi_over_mphi[i]),Fchi[i]



ell = corner.corner(Matrix_fermionic_prior,smooth = 1.0,color = 'xkcd:grey',range = [(2,8),(-5,3),(0,5.)],
                   plot_datapoints = False,plot_density = True,plot_contours = True,divergences = False, levels=(0.393, 0.864),
                    hist_kwargs = {'linestyle': '-','linewidth': 2.0}, contour_kwargs = {'linestyles':'solid','linewidths': 2.0})


figure = corner.corner(Matrix_fermionic,smooth = 1.0,fig = ell,color = c_fermionic,labels = [r"log$_{10}$(m$_\chi$/MeV)",r"log$_{10}$($\frac{\mathdefault{g}_\chi}{\mathdefault{m}_\phi/\mathrm{MeV}})$",r"F$_\chi \, [\%]$"],
                      range = [(2,8),(-5,3),(0,5.)], levels=(0.393, 0.864), show_titles = True,label_kwargs = {"fontsize":18,"font":'serif'},title_kwargs = {"fontsize":15},
                      hist_kwargs = {'linestyle': 'dotted','linewidth': 2.0}, contour_kwargs = {'linestyles':'dotted','linewidths': 2.0})



figure.subplots_adjust(right=1.15,top=1.15)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=15) 
    
figure.legend(handles =[matplotlib.lines.Line2D([],[],color = 'xkcd:grey',label = 'Fermionic ADM Prior', linestyle = '-', lw=2.0),
                        matplotlib.lines.Line2D([],[],color = c_fermionic,label = 'Fermionic ADM Posterior',linestyle = 'dotted',lw = 2.0)],
                  fontsize = 22,frameon = False,loc = "upper right")



figure.savefig(plots_path / 'Posterior_corner_bosonic_fermionic_adm.pdf',bbox_inches='tight')




fig,ax = plt.subplots(figsize = (10,11))
my_fontsize=20
my_font = 'serif'
matplotlib.rc('font',family = my_font)
matplotlib.rcParams.update({'font.size': 20})

fermionic_ratio = np.log10(10**Matrix_fermionic[:,1]/10**Matrix_fermionic[:,0])

print('The fermionic ratio of gchi/mphi to mchi is given by log10(gchi/mphi/mchi), which is given in the main text of the paper. The 68% and 95% confidence intervals for this ratio are given below.')
print('Posteriors 68% ', get_quantiles(fermionic_ratio, quantiles = [0.16,0.5,0.84]) )
print('Posteriors 95% ', get_quantiles(fermionic_ratio))
plot = sns.kdeplot(x = fermionic_ratio,y = Matrix_fermionic[:,2],shade = True,cbar = False,cmap = 'Blues_r'
                   ,common_norm = True,levels=[0.05,0.32,1.],ax = ax)


fermionic_ratio_prior = np.log10(10**Matrix_fermionic_prior[:,1]/10**Matrix_fermionic_prior[:,0])
plot_prior = sns.kdeplot(x = fermionic_ratio_prior,y = Matrix_fermionic_prior[:,2],shade = False,cbar = False,colors = 'xkcd:grey',common_norm = True,levels =[0.05,0.32,1],label = "Prior", linestyles = 'solid', linewidths = 2.,ax = ax)




ax.legend(handles =[mpatches.Patch(color = c_fermionic ,label = 'Fermionic ADM Posterior')],
                  fontsize = 18,frameon = True,loc = "upper left")



ax.set_yticks(np.arange(0, 5.5,.5),minor =True)
ax.set_xticks([-9,-8,-7,-6,-5,-4,-3])
ax.set_ylim(0.0,5.5)
ax.set_ylabel(r"F$_\chi$ [$\%$]",fontsize=my_fontsize,font = my_font)
ax.set_xlabel(r"$\log_{10}(\frac{\mathdefault{g}_\chi}{\mathdefault{m}_\phi/\mathrm{MeV}}/(\mathdefault{m}_\chi/\mathrm{MeV}))$",fontsize=my_fontsize,font = my_font)
ax.set_xlim(-9,-3)
ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)





fig.savefig(plots_path / 'fermionic_adm_ratio_plot.pdf',bbox_inches='tight')