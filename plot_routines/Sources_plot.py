import numpy as np
import matplotlib as plt
from matplotlib import pyplot


from matplotlib.patches import Ellipse 
import os
import pathlib
from pathlib import Path
import argparse
# In[3]:


plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

pyplot.rcParams['xtick.direction'] = 'in'
pyplot.rcParams['xtick.minor.visible'] = True
pyplot.rcParams['ytick.direction'] = 'in'
pyplot.rcParams['ytick.minor.visible'] = True
pyplot.rcParams['xtick.major.size'] = 5
pyplot.rcParams['ytick.major.size'] = 5
pyplot.rcParams['ytick.right'] = True
pyplot.rcParams['xtick.top'] = True 



script_dir = Path(__file__).resolve().parent

plots_directory = script_dir.parent / 'plots' if not args.repro else script_dir.parent / 'repro'/ 'plots'
plots_directory.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist


data_directory = script_dir.parent / 'data'



parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repro', action='store_true')
args = parser.parse_args()


    


# In[9]:


J0030_V24_STPDT_MR_95 = np.loadtxt(data_directory + f'J0030_Vinciguerra24_STPDT_NxXMM_95.txt')
J0030_V24_STPDT_M_95 = J0030_V24_STPDT_MR_95[:,1]
J0030_V24_STPDT_R_95 = J0030_V24_STPDT_MR_95[:,0]

J0030_V24_STPDT_MR_68 = np.loadtxt(data_directory + f'J0030_Vinciguerra24_STPDT_NxXMM_68.txt')
J0030_V24_STPDT_M_68 = J0030_V24_STPDT_MR_68[:,1]
J0030_V24_STPDT_R_68 = J0030_V24_STPDT_MR_68[:,0]


# In[10]:


J0740_S24_MR_68 = np.loadtxt(data_directory + f'J0740_Salmi24_68.txt')
J0740_S24_MR_95 = np.loadtxt(data_directory + f'J0740_Salmi24_95.txt')

J0740_S24_M_68 = J0740_S24_MR_68[:,1]
J0740_S24_R_68 = J0740_S24_MR_68[:,0]
J0740_S24_M_95 = J0740_S24_MR_95[:,1]
J0740_S24_R_95 = J0740_S24_MR_95[:,0]


# In[12]:


J0437_C24_CSTPDT_MR_68 = np.load(data_directory + f'J0437_CST_PDT_3C50_BKG_AGN_smooth_hiMN_lowXPSI_res_3sigma_68_contours.npy')
J0437_C24_CSTPDT_MR_95 = np.load(data_directory + f'J0437_CST_PDT_3C50_BKG_AGN_smooth_hiMN_lowXPSI_res_3sigma_95_contours.npy')

J0437_C24_CSTPDT_M_68 = J0437_C24_CSTPDT_MR_68[:,1]
J0437_C24_CSTPDT_R_68 = J0437_C24_CSTPDT_MR_68[:,0]
J0437_C24_CSTPDT_M_95 = J0437_C24_CSTPDT_MR_95[:,1]
J0437_C24_CSTPDT_R_95 = J0437_C24_CSTPDT_MR_95[:,0]


# In[16]:


fig, axes = pyplot.subplots(1, 1, figsize=(12,10))

ax_baseline = axes
ax_baseline.annotate('J0437', xy=(9.25, 1.4), color='xkcd:light red',fontsize = 24)
ax_baseline.annotate('J0030', xy=(11.5, 1.6), color='xkcd:aqua',fontsize = 24)
ax_baseline.annotate('J0740', xy=(11.0, 2.25), color='xkcd:light violet',fontsize = 24)
ax_baseline.tick_params(top = 1, right = 1, which="both", axis="both", direction="in",labelsize=22 )

xlim = (8, 16)
ylim = (1.0, 2.4)
ax_baseline.set_xlim(xlim)
ax_baseline.set_ylim(ylim)

# Ticks
xticks = [8,9,10,11, 12,13, 14,15, 16]
yticks = [1.0, 1.4, 1.8, 2.2]
ax_baseline.set_xticks(xticks)
ax_baseline.set_yticks(yticks)
ax_baseline.minorticks_on()


#axes.set_title(r'REAL DATA',fontsize = 28)
axes.tick_params(axis='both', which='major', labelsize=22)
axes.set_ylabel(r'$\mathrm{M \, [M_\odot]}$',fontsize = 28)
axes.set_xlabel(r'$R\, \mathrm{[km]}$',fontsize = 28)

ax_baseline.fill(J0030_V24_STPDT_R_95, J0030_V24_STPDT_M_95, linewidth=2.0, color='xkcd:aqua', alpha=0.3)
ax_baseline.fill(J0030_V24_STPDT_R_68, J0030_V24_STPDT_M_68, linewidth=2.0, color='xkcd:aqua', alpha=0.6)

ax_baseline.fill(J0437_C24_CSTPDT_R_95, J0437_C24_CSTPDT_M_95, linewidth=1.0, color='xkcd:light red',  alpha=0.2)
ax_baseline.fill(J0437_C24_CSTPDT_R_68, J0437_C24_CSTPDT_M_68, linewidth=1.0, color='xkcd:light red',  alpha=0.4)


ax_baseline.fill(J0740_S24_R_95, J0740_S24_M_95, linewidth=2.0, color='xkcd:light violet', alpha=0.4)
ax_baseline.fill(J0740_S24_R_68, J0740_S24_M_68, linewidth=2.0, color='xkcd:light violet', alpha=0.8)
fig.savefig(plots_directory + 'MRdata.pdf',bbox_inches='tight')


