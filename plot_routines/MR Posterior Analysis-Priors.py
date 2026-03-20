#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import pyplot
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm
from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib.colors import ListedColormap
import plotting


# In[2]:


import global_imports
c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s
pi = global_imports._pi
rho_ns = global_imports._rhons


# In[3]:


de = True

eps_min = 10**(14.2)/rho_ns
eps_max = 10**(16)/rho_ns
energydensities = np.linspace(eps_min,eps_max,50)*rho_ns if de else np.logspace(14,16,50)

print(energydensities)


# In[4]:


de = False

eps_min = 10**(14.2)/rho_ns
eps_max = 10**(16)/rho_ns
energydensities = np.linspace(eps_min,eps_max,50)*rho_ns if de else np.logspace(14.2,16,50)

print(energydensities)


# In[5]:


np.log10(2*rho_ns)


# In[6]:


np.log10(1.5*rho_ns)


# In[7]:


tmp_color = sns.cubehelix_palette(8, start=.5, rot=-.75, dark=0.2, light=.85)[0::3]
c_baryonic = tmp_color[:2]
c_DE = ['#E76F51', 'xkcd:tomato']
c_bADM = ['xkcd:violet', 'xkcd:violet']
c_fADM = ['xkcd:azure', 'xkcd:cobalt blue']


# In[8]:


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


# In[9]:


def calc_bands(x, y):
    miny = numpy.zeros((len(y),3))
    maxy = numpy.zeros((len(y),3))
    
    for i in tqdm(range(len(y))):
        z = y[i][y[i]>0.0]
        if len(z)<200:
            print('sample too small for %.2f' %x[i])
            continue
        kde = gaussian_kde(z)
        testz = numpy.linspace(min(z),max(z), 1000)
        pdf = kde.pdf(testz)
        array = pdf
        index_68 = numpy.where(numpy.cumsum(numpy.sort(array)[::-1]) < sum(array)*0.6827)[0]
        index_68 = numpy.argsort(array)[::-1][index_68]
        index_95 = numpy.where(numpy.cumsum(numpy.sort(array)[::-1]) < sum(array)*0.95)[0]
        index_95 = numpy.argsort(array)[::-1][index_95]
        miny[i] =  x[i], min(testz[index_68]), min(testz[index_95])
        maxy[i] =  x[i], max(testz[index_68]), max(testz[index_95])
        
    miny = miny[~numpy.all(miny == 0, axis=1)]
    maxy = maxy[~numpy.all(maxy == 0, axis=1)]
    return miny, maxy


# In[10]:


Baryonic_pressures_prior = numpy.load('B/Baryonic_prior_pressures.npy')

energydensities = numpy.logspace(14.2, 16, 50)
Baryonic_pr_contours = calc_bands(energydensities,Baryonic_pressures_prior)


# minpres_prior = np.load('B/Baryonic_prior_minpres.npy')
# maxpres_prior = np.load('B/Baryonic_prior_maxpres.npy')

# Baryonic_pr_contours = minpres_prior, maxpres_prior


# In[11]:


energydensities = numpy.logspace(14.2, 16, 50)
#pressures_prior = numpy.load('FERMIONIC_REAL_DATA_PRIOR_Pressure_array.npy')
minpres_prior = np.load('DE/Dark_energy_prior_minpres_total.npy')
maxpres_prior = np.load('DE/Dark_energy_prior_maxpres_total.npy')


DE_prior_contours = minpres_prior,maxpres_prior
#prior_contours = calc_bands(energydensities, pressures_prior)
print(numpy.shape(DE_prior_contours))



energydensities_b = np.logspace(14.2, 16, 250)
energydensities_dm = np.logspace(11, 18, 250)
energydensities_ADM = energydensities_b + energydensities_dm

Bosonic_minpres_pr = numpy.load('BDM/Bosonic_prior_minpres.npy')
Bosonic_maxpres_pr = numpy.load('BDM/Bosonic_prior_maxpres.npy')
Bosonic_pr_contours = Bosonic_minpres_pr,Bosonic_maxpres_pr

# pressures = numpy.load('BDM/Bosonic_posterior_pressures.npy')
# Bosonic_posterior_contours = calc_bands(energydensities_ADM, pressures)

print(numpy.shape(Bosonic_pr_contours))


Fermionic_minpres_pr = numpy.load('FDM/Fermionic_prior_minpres.npy')
Fermionic_maxpres_pr= numpy.load('FDM/Fermionic_prior_maxpres.npy')
Fermionic_pr_contours = Fermionic_minpres_pr,Fermionic_maxpres_pr


# In[12]:


FADM_maxpres = numpy.load('FDM/Fermionic_prior_maxpres.npy')


maxpres_FADM_interp = UnivariateSpline(FADM_maxpres[:,0], FADM_maxpres[:,2], k=1, s = 0, ext = 1)

FADM_maxpres_baryonic = numpy.load('FDM/Fermionic_prior_maxpres_baryon.npy')

maxpres_FADM_baryonic_interp = UnivariateSpline(FADM_maxpres_baryonic[:,0], FADM_maxpres_baryonic[:,2], k=1, s = 0, ext = 1)


BADM_maxpres = numpy.load('BDM/Bosonic_prior_maxpres.npy')

maxpres_BADM_interp = UnivariateSpline(BADM_maxpres[:,0], BADM_maxpres[:,2], k=1, s = 0, ext = 1)

BADM_maxpres_baryonic = numpy.load('BDM/Bosonic_prior_maxpres_baryon.npy')

maxpres_BADM_baryonic_interp = UnivariateSpline(BADM_maxpres_baryonic[:,0], BADM_maxpres_baryonic[:,2], k=1, s = 0, ext = 1)


# In[13]:


#Baryonic_pr_contours #miny (x, min_68, min, 95); maxy (x, max 68, max 95)
maxpres_Baryonic = numpy.log10(Baryonic_pr_contours[1])


maxpres_Baryonic_interp = UnivariateSpline(maxpres_Baryonic[:,0], maxpres_Baryonic[:,2], k=1, s = 0, ext = 1)
print(maxpres_Baryonic_interp(14.4), maxpres_Baryonic_interp(14.85))
#print(maxpres_Baryonic[:,2][8],maxpres_Baryonic[:,2][18])


# In[14]:


# print(np.log10(FADM_maxpres[:,0][42]))
# print(np.log10(FADM_maxpres[:,0][90]))

press_FADM_dm = maxpres_FADM_interp(10**14.4) - maxpres_FADM_baryonic_interp(10**14.4)   #np.log10(FADM_maxpres[:,2][42] - FADM_maxpres_baryonic[:,2][42])

print(np.log10(press_FADM_dm), np.log10(maxpres_FADM_baryonic_interp(10**14.4)))
print(press_FADM_dm/pow(10,maxpres_Baryonic_interp(14.4))*100)

press_FADM_dm = maxpres_FADM_interp(10**14.85) - maxpres_FADM_baryonic_interp(10**14.85)
print(np.log10(press_FADM_dm), np.log10(maxpres_FADM_baryonic_interp(10**14.85)))

print(press_FADM_dm/pow(10,maxpres_Baryonic_interp(14.85))*100)

press_BADM_dm = maxpres_BADM_interp(10**14.4) - maxpres_BADM_baryonic_interp(10**14.4)
print(np.log10(press_BADM_dm), np.log10(maxpres_BADM_baryonic_interp(10**14.4)))

print(press_BADM_dm/pow(10,maxpres_Baryonic_interp(14.4))*100)

press_BADM_dm = maxpres_BADM_interp(10**14.85) - maxpres_BADM_baryonic_interp(10**14.85)
print(np.log10(press_BADM_dm), np.log10(maxpres_BADM_baryonic_interp(10**14.85)))

print(press_BADM_dm/pow(10,maxpres_Baryonic_interp(14.85))*100)

#print(np.log10(BADM_maxpres_baryonic[:,2][83]), np.log10(BADM_maxpres[:,2][83]))


# pressures = numpy.load('POSTERIOR_BARYONIC_REAL_pressures.npy')
# contours = calc_bands(energydensities, pressures)
# contours_min = contours[0]
# contours_max = contours[1]
# minpres_ppNI = numpy.log10(contours_min)
# maxpres_ppNI = numpy.log10(contours_max)

# In[ ]:





# In[ ]:





# In[15]:


def mass_radius_prior_plot(root_name_1,root_name_2,root_name_3, root_name_4,ax = None):
    scatter_1 = numpy.loadtxt(root_name_1 + 'MR_prpr.txt')
    # scatter_ADM = []
    # for i in range(len(pre_scatter_ADM)):
    #     if pre_scatter_ADM[i][1] >0. and pre_scatter_ADM[i][1] < 20.: #eliminating Halos (>0) and the few stars that were sampled of having radii larger than what NICER considers (>20)
    #         scatter_ADM.append([pre_scatter_ADM[i][0],pre_scatter_ADM[i][1]])
            
    # scatter_ADM = numpy.array(scatter_ADM)
    

    
    sns.kdeplot(x = scatter_1[:,1], y = scatter_1[:,0], gridsize=40, 
               fill=False, ax=ax, levels=[0.05,1.],bw_adjust = 1.5,
                alpha=1., colors = c_DE,linestyles = '-.',linewidths = 3.)

    
    scatter_2 = numpy.loadtxt(root_name_2 + 'MR_prpr.txt')
        # scatter_prior = []
        # for i in range(len(pre_scatter_prior)):
        #     if pre_scatter_prior[i][-1] ==0.: #eliminating any halos that might appear after resampling the central density
        #         scatter_prior.append(pre_scatter_prior[i]) 
            
        # scatter_prior = numpy.array(scatter_prior)


    
    sns.kdeplot(x = scatter_2[:,1], y = scatter_2[:,0], gridsize=40, 
                fill=False, ax=ax, levels=[0.05,1.],bw_adjust = 1.5,
                alpha=1., colors = c_fADM,linestyles = 'dotted',linewidths = 3.)

    scatter_3= numpy.loadtxt(root_name_3 + 'MR_prpr.txt')

    
    sns.kdeplot(x = scatter_3[:,1], y = scatter_3[:,0], gridsize=40,bw_adjust = 1.5, 
                fill=True, ax=ax, levels=[0.05,0.32,1.],
                alpha=1., cmap=ListedColormap(c_baryonic))

    # if root_name_4 is not None:
    scatter_4 = numpy.loadtxt(root_name_4 + 'MR_prpr.txt')
    sns.kdeplot(x = scatter_4[:,1], y = scatter_4[:,0], gridsize=40, 
                fill=False, ax=ax, levels=[0.05,1.],bw_adjust = 1.5,
                alpha=1., colors = c_bADM,linestyles = '--',linewidths = 3.)
        
    
    #custom_lines = [Line2D([0],[0],color='#E76F51', alpha=1.,linestyle = '-.',lw = 3.),mpatches.Patch(color=sns.cubehelix_palette(8, start=.5, rot=-.75, dark=.2, light=.85)[0::3][1])]
   # ax.legend(custom_lines, ['Including ADM','Neglecting ADM'],
                #loc='upper left', prop={'size': 16})
    
    ax.set_xlim(9, 15)
    ax.set_xticks([10,11,12,13,14,15])
    ax.set_ylim(1., 2.7)
    ax.set_yticks([1.,1.4,1.8,2.2,2.7])
    
    #ax.set_title('Mass-Radius Posteriors',font = 'serif',fontsize = 24)
    ax.minorticks_on()
    ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=20)
    ax.set_xlabel(r'Radius [km]', fontsize=20)
    ax.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=20)
    
    #figure.savefig('MRpost_real_data.png',bbox_inches='tight')


# In[16]:


root_name_1 = 'DE/Dark_energy_prior_'
root_name_2 = 'FDM/Fermionic_prior_'
root_name_3 = 'B/Baryonic_prior_'
root_name_4 = 'BDM/Bosonic_prior_'


# In[17]:


minpres_ppNI = numpy.log10(Baryonic_pr_contours[0])
maxpres_ppNI = numpy.log10(Baryonic_pr_contours[1])


# In[18]:


fig, ax = pyplot.subplots(nrows=1, ncols=2,figsize=(13,6))
fig.subplots_adjust(wspace=0.5, hspace=0)
my_fontsize=18
my_font = 'serif'
matplotlib.rc('font',family = my_font)
matplotlib.rcParams.update({'font.size': 18})

mass_radius_prior_plot(root_name_1,root_name_2,root_name_3,root_name_4,ax = ax[1])




# posterior_contours_min = posterior_contours[0]
# posterior_contours_max = posterior_contours[1]
# minpres_posterior = numpy.log10(posterior_contours_min)
# maxpres_posterior = numpy.log10(posterior_contours_max)

Bosonic_pr_contours_min = Bosonic_pr_contours[0]
Bosonic_pr_contours_max = Bosonic_pr_contours[1]
Bosonic_minpres_pr = numpy.log10(Bosonic_pr_contours_min)
Bosonic_maxpres_pr = numpy.log10(Bosonic_pr_contours_max)



Fermionic_pr_contours_min = Fermionic_pr_contours[0]
Fermionic_pr_contours_max = Fermionic_pr_contours[1]
Fermionic_minpres_pr = numpy.log10(Fermionic_pr_contours_min)
Fermionic_maxpres_pr = numpy.log10(Fermionic_pr_contours_max)

prior_contours_min = DE_prior_contours[0]
prior_contours_max = DE_prior_contours[1]
DE_minpres_pr = numpy.log10(prior_contours_min)
DE_maxpres_pr = numpy.log10(prior_contours_max)




# ax[0].plot(maxpres_prior[:,0], minpres_prior[:,2], c='black', linestyle='--', lw=2.75)
# ax[0].plot(maxpres_prior[:,0], maxpres_prior[:,2], c='black', linestyle='--', lw=2.75,label = 'Prior')

# ax[0].plot(Bosonic_maxpres_pr[:,0], Bosonic_minpres_pr[:,1], c=c_bADM[1], linestyle='--', lw=4)
# ax[0].plot(Bosonic_maxpres_pr[:,0], Bosonic_maxpres_pr[:,1], c=c_bADM[1], linestyle='--', lw=4)

ax[0].plot(Bosonic_maxpres_pr[:,0], Bosonic_minpres_pr[:,2], c=c_bADM[1], linestyle='--', lw=4)
ax[0].plot(Bosonic_maxpres_pr[:,0], Bosonic_maxpres_pr[:,2], c=c_bADM[1], linestyle='--', lw=4)

# ax[0].plot(Fermionic_maxpres_pr[:,0], Fermionic_minpres_pr[:,1], c=c_fADM[1], linestyle='dotted', lw=4)
# ax[0].plot(Fermionic_maxpres_pr[:,0], Fermionic_maxpres_pr[:,1], c=c_fADM[1],
#                                                         linestyle='dotted', lw=4)

ax[0].plot(Fermionic_maxpres_pr[:,0], Fermionic_minpres_pr[:,2], c=c_fADM[0], linestyle='dotted', lw=4)
ax[0].plot(Fermionic_maxpres_pr[:,0], Fermionic_maxpres_pr[:,2], c=c_fADM[0],
                                                        linestyle='dotted', lw=4)


# ax[0].plot(DE_maxpres_pr[:,0], DE_minpres_pr[:,1], c=c_DE[1], linestyle='-.', lw=4)
# ax[0].plot(DE_maxpres_pr[:,0], DE_maxpres_pr[:,1], c=c_DE[1],
#                                                         linestyle='-.', lw=4)

ax[0].plot(DE_maxpres_pr[:,0], DE_minpres_pr[:,2], c=c_DE[0], linestyle='-.', lw=4)
ax[0].plot(DE_maxpres_pr[:,0], DE_maxpres_pr[:,2], c=c_DE[0],
                                                        linestyle='-.', lw=4)




ax[0].fill_between(minpres_ppNI[:,0], minpres_ppNI[:,2], maxpres_ppNI[:,2], 
                       color=c_baryonic[0], alpha=1)

ax[0].fill_between(minpres_ppNI[:,0], minpres_ppNI[:,1], maxpres_ppNI[:,1], 
                       color=c_baryonic[1], 
                      alpha=1)
       

ax[0].set_ylabel(r'$\log_{10}(P)$ [dyn/cm$^2$]', size=my_fontsize,font = my_font)

masses = np.arange(1,3.2,.1)*Msun
rg3 = 3*masses*G/c**2/1e5
ax[1].plot(rg3,masses/Msun,color = 'xkcd:black')
line1 = plotting.custom_line(c_bADM[1], 'dashed', lw=4.)
line2 = plotting.custom_line(c_fADM[0], 'dotted', lw=4.)
line3 = plotting.custom_line(c_DE[0], '-.', lw=4.)
line4 = plotting.double_interval_legend(c_baryonic)
line5 = plotting.custom_line('xkcd:black', )
custom_lines = [line1, line2, line3, line4, line5]
ax[0].legend(custom_lines, ['Bosonic ADM + PP EoS', 'Fermionic ADM + PP EoS', 'MCDF + PP EoS','PP EoS', 'Compactness line'],loc = 'upper left',fontsize = 16, frameon=False)

ax[0].set_xlim(14.25, 15.24)
ax[0].set_xlabel(r'$\log_{10}(\epsilon)$ [g/cm$^3$]', fontsize=my_fontsize,font = my_font)
ax[0].set_xticks(numpy.arange(14.25, 15.24,.05),minor =True)
#ax[0].set_ylim(33,36)
#ax[0].set_yticks([33,33.5,34,34.5,35,35.5,36],minor =True)
ax[0].tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)

ax[1].set_xlim(6, 16)
ax[1].set_xticks([6,7,8,9,10,11,12,13,14,15,16])
ax[1].set_ylim(1., 3.1)
ax[1].set_yticks([1.,1.4,1.8,2.2,2.6,3.0])
    
    #ax.set_title('Mass-Radius Posteriors',font = 'serif',fontsize = 24)
ax[1].minorticks_on()
ax[1].tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)
ax[1].set_xlabel(r'$R$ [km]', fontsize=my_fontsize)
ax[1].set_ylabel(r'M [M$_{\odot}$]', fontsize=my_fontsize)
    

pyplot.tight_layout()
pyplot.show()
fig.savefig('plots/EoS_MR_priors_total.pdf',bbox_inches='tight')


# In[ ]:





# In[19]:


energydensities = numpy.logspace(14.2, 16, 50)
#pressures_prior = numpy.load('FERMIONIC_REAL_DATA_PRIOR_Pressure_array.npy')
minpres_prior = np.load('DE/Dark_energy_prior_minpres_baryon.npy')
maxpres_prior = np.load('DE/Dark_energy_prior_maxpres_baryon.npy')


DE_prior_contours = minpres_prior,maxpres_prior
#prior_contours = calc_bands(energydensities, pressures_prior)
print(numpy.shape(DE_prior_contours))


Bosonic_minpres_pr = numpy.load('BDM/Bosonic_prior_minpres_baryon.npy')
Bosonic_maxpres_pr = numpy.load('BDM/Bosonic_prior_maxpres_baryon.npy')
Bosonic_pr_contours = Bosonic_minpres_pr,Bosonic_maxpres_pr

# pressures = numpy.load('BDM/Bosonic_posterior_pressures.npy')
# Bosonic_posterior_contours = calc_bands(energydensities_ADM, pressures)

print(numpy.shape(Bosonic_pr_contours))


Fermionic_minpres_pr = numpy.load('FDM/Fermionic_prior_minpres_baryon.npy')
Fermionic_maxpres_pr= numpy.load('FDM/Fermionic_prior_maxpres_baryon.npy')
Fermionic_pr_contours = Fermionic_minpres_pr,Fermionic_maxpres_pr


# In[ ]:





# In[20]:


fig, ax = pyplot.subplots(nrows=1, ncols=1,figsize=(8,6))
my_fontsize=18
my_font = 'serif'
matplotlib.rc('font',family = my_font)
matplotlib.rcParams.update({'font.size': 18})




Bosonic_pr_contours_min = Bosonic_pr_contours[0]
Bosonic_pr_contours_max = Bosonic_pr_contours[1]
Bosonic_minpres_pr = numpy.log10(Bosonic_pr_contours_min)
Bosonic_maxpres_pr = numpy.log10(Bosonic_pr_contours_max)



Fermionic_pr_contours_min = Fermionic_pr_contours[0]
Fermionic_pr_contours_max = Fermionic_pr_contours[1]
Fermionic_minpres_pr = numpy.log10(Fermionic_pr_contours_min)
Fermionic_maxpres_pr = numpy.log10(Fermionic_pr_contours_max)

prior_contours_min = DE_prior_contours[0]
prior_contours_max = DE_prior_contours[1]
DE_minpres_pr = numpy.log10(prior_contours_min)
DE_maxpres_pr = numpy.log10(prior_contours_max)




# ax.plot(Bosonic_maxpres_pr[:,0], Bosonic_minpres_pr[:,1], c=c_bADM[1], linestyle='--', lw=4)
# ax.plot(Bosonic_maxpres_pr[:,0], Bosonic_maxpres_pr[:,1], c=c_bADM[1], linestyle='--', lw=4)

ax.plot(Bosonic_maxpres_pr[:,0], Bosonic_minpres_pr[:,2], c=c_bADM[1], linestyle='--', lw=4)
ax.plot(Bosonic_maxpres_pr[:,0], Bosonic_maxpres_pr[:,2], c=c_bADM[1], linestyle='--', lw=4,label = 'Including Bosonic ADM')

# ax.plot(Fermionic_maxpres_pr[:,0], Fermionic_minpres_pr[:,1], c=c_fADM[1], linestyle='dotted', lw=4)
# ax.plot(Fermionic_maxpres_pr[:,0], Fermionic_maxpres_pr[:,1], c=c_fADM[1],
#                                                         linestyle='dotted', lw=4)

ax.plot(Fermionic_maxpres_pr[:,0], Fermionic_minpres_pr[:,2], c=c_fADM[0], linestyle='dotted', lw=4)
ax.plot(Fermionic_maxpres_pr[:,0], Fermionic_maxpres_pr[:,2], c=c_fADM[0],
                                                        linestyle='dotted', lw=4,label = 'Including Fermionic ADM')


# ax.plot(DE_maxpres_pr[:,0], DE_minpres_pr[:,1], c=c_DE[1], linestyle='-.', lw=4)
# ax.plot(DE_maxpres_pr[:,0], DE_maxpres_pr[:,1], c=c_DE[1],
#                                                         linestyle='-.', lw=4)

ax.plot(DE_maxpres_pr[:,0], DE_minpres_pr[:,2], c=c_DE[0], linestyle='-.', lw=4)
ax.plot(DE_maxpres_pr[:,0], DE_maxpres_pr[:,2], c=c_DE[0],
                                                        linestyle='-.', lw=4, label = 'Including MCDF')




ax.fill_between(minpres_ppNI[:,0], minpres_ppNI[:,2], maxpres_ppNI[:,2], 
                       color=c_baryonic[0], alpha=1)

ax.fill_between(minpres_ppNI[:,0], minpres_ppNI[:,1], maxpres_ppNI[:,1], 
                       color=c_baryonic[1], 
                      alpha=1,label = 'Baryonic EoS')
       

ax.set_ylabel(r'$\log_{10}(P)$ [dyn/cm$^2$]', size=my_fontsize,font = my_font)



ax.legend(loc = 'upper left',fontsize = 14,frameon=False)
ax.set_xlim(14.25, 15.24)
ax.set_xlabel(r'$\log_{10}(\epsilon)$ [g/cm$^3$]', fontsize=my_fontsize,font = my_font)
ax.set_xticks(numpy.arange(14.25, 15.24,.05),minor =True)
ax.set_ylim(33,36)
ax.set_yticks([33,33.5,34,34.5,35,35.5,36],minor =True)
ax.tick_params(top=1,right=1, which='both', direction='in', labelsize=my_fontsize)

    
    
pyplot.tight_layout()
pyplot.show()
fig.savefig('plots/EoS_priors_baryon.pdf',bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[ ]:




