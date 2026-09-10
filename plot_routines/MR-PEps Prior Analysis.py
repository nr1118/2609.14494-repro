
import matplotlib
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm
from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib.colors import ListedColormap
import os
import pathlib
from pathlib import Path

import argparse


import plotting


# In[2]:

import global_imports
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()

# In[7]:
script_dir = Path(__file__).resolve().parent

plots_path = script_dir.parent / 'plots' if not args.repro else script_dir.parent / 'repro'/ 'plots'
plots_path.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist


tmp_color = sns.cubehelix_palette(8, start=.5, rot=-.75, dark=0.2, light=.85)[0::3]
c_baryonic = tmp_color[:2]
c_DE = ['#E76F51', 'xkcd:tomato']
c_bADM = ['xkcd:violet', 'xkcd:violet']
c_fADM = ['xkcd:azure', 'xkcd:cobalt blue']


# In[8]:


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


# In[9]:


def calc_bands(x, y):
    miny = np.zeros((len(y),3))
    maxy = np.zeros((len(y),3))
    
    for i in tqdm(range(len(y))):
        z = y[i][y[i]>0.0]
        if len(z)<200:
            print('sample too small for %.2f' %x[i])
            continue
        kde = gaussian_kde(z)
        testz = np.linspace(min(z),max(z), 1000)
        pdf = kde.pdf(testz)
        array = pdf
        index_68 = np.where(np.cumsum(np.sort(array)[::-1]) < sum(array)*0.6827)[0]
        index_68 = np.argsort(array)[::-1][index_68]
        index_95 = np.where(np.cumsum(np.sort(array)[::-1]) < sum(array)*0.95)[0]
        index_95 = np.argsort(array)[::-1][index_95]
        miny[i] =  x[i], min(testz[index_68]), min(testz[index_95])
        maxy[i] =  x[i], max(testz[index_68]), max(testz[index_95])
        
    miny = miny[~np.all(miny == 0, axis=1)]
    maxy = maxy[~np.all(maxy == 0, axis=1)]
    return miny, maxy


# In[10]:







def mass_radius_prior_plot(root_name_DE,root_name_FDM,root_name_B, root_name_BDM,ax = None):
    scatter_1 = np.loadtxt(root_name_DE + 'MR_prpr.txt')

    
    
    sns.kdeplot(x = scatter_1[:,1], y = scatter_1[:,0], gridsize=40, 
               fill=False, ax=ax, levels=[0.05,1.],bw_adjust = 1.5,
                alpha=1., colors = c_DE,linestyles = '-.',linewidths = 3.)

    
    scatter_2 = np.loadtxt(root_name_FDM + 'MR_prpr.txt')



    
    sns.kdeplot(x = scatter_2[:,1], y = scatter_2[:,0], gridsize=40, 
                fill=False, ax=ax, levels=[0.05,1.],bw_adjust = 1.5,
                alpha=1., colors = c_fADM,linestyles = 'dotted',linewidths = 3.)

    scatter_3= np.loadtxt(root_name_B + 'MR_prpr.txt')

    
    sns.kdeplot(x = scatter_3[:,1], y = scatter_3[:,0], gridsize=40,bw_adjust = 1.5, 
                fill=True, ax=ax, levels=[0.05,0.32,1.],
                alpha=1., cmap=ListedColormap(c_baryonic))

    # if root_name_4 is not None:
    scatter_4 = np.loadtxt(root_name_BDM + 'MR_prpr.txt')
    sns.kdeplot(x = scatter_4[:,1], y = scatter_4[:,0], gridsize=40, 
                fill=False, ax=ax, levels=[0.05,1.],bw_adjust = 1.5,
                alpha=1., colors = c_bADM,linestyles = '--',linewidths = 3.)
        
    
    
    ax.set_xlim(9, 15)
    ax.set_xticks([10,11,12,13,14,15])
    ax.set_ylim(1., 2.7)
    ax.set_yticks([1.,1.4,1.8,2.2,2.7])
    
    ax.minorticks_on()
    ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=20)
    ax.set_xlabel(r'Radius [km]', fontsize=20)
    ax.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=20)
    







my_fontsize = 18
my_font = 'serif'
matplotlib.rc('font', family=my_font)
matplotlib.rcParams.update({'font.size': 18})

# Create a 1x3 figure layout
fig = plt.figure(figsize=(18, 14))
gs = matplotlib.gridspec.GridSpec(2, 4, figure=fig, wspace=0.6, hspace=0.3)


ax = [
    fig.add_subplot(gs[0, 0:2]), # Top left
    fig.add_subplot(gs[0, 2:4]), # Top right 
    fig.add_subplot(gs[1, 1:3])  # Bottom centered: spans cols 1 and 2
]

root_name_DE = script_dir /'results'/ 'prior' / 'DE' / 'Dark_energy_prior_'
root_name_BDM = script_dir /'results'/ 'prior' / 'BDM' / 'Bosonic_prior_'
root_name_B = script_dir / 'results' / 'prior' / 'B' / 'Baryonic_prior_'
root_name_FDM = script_dir / 'results' / 'prior' / 'FDM' / 'Fermionic_prior_'


Baryonic_pressures_prior = np.load(root_name_B + 'pressures.npy')

energydensities = np.logspace(14.2, 16, 50) #Taken from PosteriorAnalysis.py for the baryonic case
Baryonic_pr_contours = calc_bands(energydensities,Baryonic_pressures_prior)
B_minpres = np.log10(Baryonic_pr_contours[0])
B_maxpres = np.log10(Baryonic_pr_contours[1])

# Load Total EoS Arrays (from previous steps)
DE_min_tot = np.log10(np.load(root_name_DE + 'minpres_total.npy'))
DE_max_tot = np.log10(np.load(root_name_DE + 'maxpres_total.npy'))
Bos_min_tot = np.log10(np.load(root_name_BDM + 'minpres.npy'))
Bos_max_tot = np.log10(np.load(root_name_BDM + 'maxpres.npy'))
Ferm_min_tot = np.log10(np.load(root_name_FDM + 'minpres.npy'))
Ferm_max_tot = np.log10(np.load(root_name_FDM + 'maxpres.npy'))

# Load Baryonic EoS Arrays
DE_min_bar = np.log10(np.load(root_name_DE + 'minpres_baryon.npy'))
DE_max_bar = np.log10(np.load(root_name_DE + 'maxpres_baryon.npy'))
Bos_min_bar = np.log10(np.load(root_name_BDM + 'minpres_baryon.npy'))
Bos_max_bar = np.log10(np.load(root_name_BDM + 'maxpres_baryon.npy'))
Ferm_min_bar = np.log10(np.load(root_name_FDM + 'minpres_baryon.npy'))
Ferm_max_bar = np.log10(np.load(root_name_FDM + 'maxpres_baryon.npy'))




# ==========================================
#  Total Combined EoS in pressure-energy density space
# ==========================================
ax[0].plot(Bos_max_tot[:,0], Bos_min_tot[:,2], c=c_bADM[1], linestyle='--', lw=4)
ax[0].plot(Bos_max_tot[:,0], Bos_max_tot[:,2], c=c_bADM[1], linestyle='--', lw=4)

ax[0].plot(Ferm_max_tot[:,0], Ferm_min_tot[:,2], c=c_fADM[0], linestyle='dotted', lw=4)
ax[0].plot(Ferm_max_tot[:,0], Ferm_max_tot[:,2], c=c_fADM[0], linestyle='dotted', lw=4)

ax[0].plot(DE_max_tot[:,0], DE_min_tot[:,2], c=c_DE[0], linestyle='-.', lw=4)
ax[0].plot(DE_max_tot[:,0], DE_max_tot[:,2], c=c_DE[0], linestyle='-.', lw=4)

ax[0].fill_between(B_minpres[:,0], B_minpres[:,2], B_maxpres[:,2], color=c_baryonic[0], alpha=1)
ax[0].fill_between(B_minpres[:,0], B_minpres[:,1], B_maxpres[:,1], color=c_baryonic[1], alpha=1)

ax[0].set_ylabel(r'$\log_{10}(P)$ [dyn/cm$^2$]', size=my_fontsize, font=my_font)
ax[0].set_xlim(14.25, 15.24)
ax[0].set_xlabel(r'$\log_{10}(\epsilon)$ [g/cm$^3$]', fontsize=my_fontsize, font=my_font)
ax[0].set_xticks(np.arange(14.25, 15.24, .05), minor=True)
ax[0].set_ylim(33, 36)
ax[0].set_yticks([33, 33.5, 34, 34.5, 35, 35.5, 36], minor=True)
ax[0].tick_params(top=1, right=1, which='both', direction='in', labelsize=my_fontsize)
ax[0].legend(loc='upper left', fontsize=18, frameon=False)

# Add custom legend for the middle panel
line1 = plotting.custom_line(c_bADM[1], 'dashed', lw=4.)
line2 = plotting.custom_line(c_fADM[0], 'dotted', lw=4.)
line3 = plotting.custom_line(c_DE[0], '-.', lw=4.)
line4 = plotting.double_interval_legend(c_baryonic)
#line5 = plotting.custom_line('xkcd:black', )
custom_lines = [line1, line2, line3, line4]
ax[0].legend(custom_lines, ['Bosonic ADM + PP EoS', 'Fermionic ADM + PP EoS', 'MCDF + PP EoS','PP EoS'], loc='upper left', fontsize=18, frameon=False)


# ==========================================
# Mass-Radius posteriors
# ==========================================
mass_radius_prior_plot(root_name_DE, root_name_FDM, root_name_B, root_name_BDM, ax=ax[1])



ax[1].set_xlim(6, 16)
ax[1].set_xticks([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
ax[1].set_ylim(1., 3.1)
ax[1].set_yticks([1., 1.4, 1.8, 2.2, 2.6, 3.0])
ax[1].minorticks_on()
ax[1].tick_params(top=1, right=1, which='both', direction='in', labelsize=my_fontsize)
ax[1].set_xlabel(r'$R$ [km]', fontsize=my_fontsize)
ax[1].set_ylabel(r'M [M$_{\odot}$]', fontsize=my_fontsize)



# ==========================================
# Baryonic EoS in pressure-energy density space
# ==========================================
ax[2].plot(Bos_max_bar[:,0], Bos_min_bar[:,2], c=c_bADM[1], linestyle='--', lw=4)
ax[2].plot(Bos_max_bar[:,0], Bos_max_bar[:,2], c=c_bADM[1], linestyle='--', lw=4, label='Neglecting Bosonic ADM')

ax[2].plot(Ferm_max_bar[:,0], Ferm_min_bar[:,2], c=c_fADM[0], linestyle='dotted', lw=4)
ax[2].plot(Ferm_max_bar[:,0], Ferm_max_bar[:,2], c=c_fADM[0], linestyle='dotted', lw=4, label='Neglecting Fermionic ADM')

ax[2].plot(DE_max_bar[:,0], DE_min_bar[:,2], c=c_DE[0], linestyle='-.', lw=4)
ax[2].plot(DE_max_bar[:,0], DE_max_bar[:,2], c=c_DE[0], linestyle='-.', lw=4, label='Neglecting MCDF')

ax[2].fill_between(B_minpres[:,0], B_minpres[:,2], B_maxpres[:,2], color=c_baryonic[0], alpha=1)
ax[2].fill_between(B_minpres[:,0], B_minpres[:,1], B_maxpres[:,1], color=c_baryonic[1], alpha=1)

ax[2].set_ylabel(r'$\log_{10}(P)$ [dyn/cm$^2$]', size=my_fontsize, font=my_font)
ax[2].set_xlim(14.25, 15.24)
ax[2].set_xlabel(r'$\log_{10}(\epsilon)$ [g/cm$^3$]', fontsize=my_fontsize, font=my_font)
ax[2].set_xticks(np.arange(14.25, 15.24, .05), minor=True)
ax[2].set_ylim(33, 36)
ax[2].set_yticks([33, 33.5, 34, 34.5, 35, 35.5, 36], minor=True)
ax[2].tick_params(top=1, right=1, which='both', direction='in', labelsize=my_fontsize)
ax[2].legend(loc='upper left', fontsize=18, frameon=False)



plt.show()
fig.savefig(plots_path / 'EoS_MR_priors_combined.pdf', bbox_inches='tight')





#Computing the maximum pressure prior distributions at the 95% confidence level of the total combined EOS (ADM + Baryonic) and the baryonic EOS alone, for both the bosonic and fermionic cases. This is done by interpolating the maximum pressure at the 95% confidence level (third column in the maxpres arrays) as a function of energy density (first column in the maxpres arrays) for each case.

densities = np.log10(np.load(root_name_BDM + 'maxpres.npy'))[:,0] #all the same densities checked over
Bos_max_tot = np.log10(np.load(root_name_BDM + 'maxpres.npy'))[:,2]
Ferm_max_tot = np.log10(np.load(root_name_FDM + 'maxpres.npy'))[:,2]
Ferm_max_bar = np.log10(np.load(root_name_FDM + 'maxpres_baryon.npy'))[:,2]
Bos_max_bar = np.log10(np.load(root_name_BDM + 'maxpres_baryon.npy'))[:,2]


log_interp_bosonic = UnivariateSpline(densities, Bos_max_tot, k=1, s=0)
log_interp_bosonic_baryonic = UnivariateSpline(densities, Bos_max_bar, k=1, s=0)

log_interp_fermionic = UnivariateSpline(densities, Ferm_max_tot, k=1, s=0)
log_interp_fermionic_baryonic = UnivariateSpline(densities, Ferm_max_bar, k=1, s=0)

maxpres_Baryonic = np.log10(Baryonic_pr_contours[1])


log_interp_Baryonic = UnivariateSpline(maxpres_Baryonic[:,0], maxpres_Baryonic[:,2], k=1, s = 0, ext = 1)



print('Baryonic pressure at 14.4 and 14.7:', log_interp_Baryonic(14.4), log_interp_Baryonic(14.7))
print('Bosonic pressure at 14.4 and 14.7:', np.log10(10**log_interp_bosonic(14.4) - 10**log_interp_bosonic_baryonic(14.4)), np.log10(10**log_interp_bosonic(14.7) - 10**log_interp_bosonic_baryonic(14.7)))
print('Fermionic pressure at 14.4 and 14.7:', np.log10(10**log_interp_fermionic(14.4) - 10**log_interp_fermionic_baryonic(14.4)), np.log10(10**log_interp_fermionic(14.7) - 10**log_interp_fermionic_baryonic(14.7)))