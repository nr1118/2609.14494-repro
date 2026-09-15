# 2609.14494-repro (Rutherford et al. 2026)
Reproduction Package for the upcoming paper, titled ``NICER neutron stars with dark energy and dark matter: effects on the inferred equation of state" by Rutherford, Prescod-Weinstein, and Watts 2026 [arXix:2609.14494](https://arxiv.org/abs/2609.14494)



NOTICE
======================
Run times range from about day for runs with posteriors which only vary the baryonic matter EoS to about 10-20 days for the priors/posteriors which additionally include the ADM EoS or Modified Chaplygin Dark Fluid (MCDF) EoS on a supercomputing cluster. While it is possible to perform prior and posterior inferences on a personal computer with mpiexec commands (for parallelization), it is NOT recommended, as the computational demand and the deterioration such runs can have on the cpu and other hardware. However, small test runs with about 100 live points can be safely run on a personal computer or laptop to ensure the script is running as desired. 

REQUIREMENTS
============
To reproduce our data and plots, the following additional requirements apply (see also NEoST installation instructions):

  - [neost version 3.0.0](https://github.com/xpsi-group/neost/tree/Dark_Energy_Adaptation). Note, this reproduction package also contains the latest version of the branch as of 9/10/26 for your convenience. See the `Dark_Energy_Adaptation` directory.
  - cython
  - pymultinest
  - gsl
  - kalepy
  - scipy
  - matplotlib
  - seaborn
  - numpy
  - corner
  - numba (optional)

Here, Numba is only needed to use the Python TOV solvers. However, normally the much faster Cython TOV solvers should be used (see [https://xpsi-group.github.io/neost/index.html](https://xpsi-group.github.io/neost/index.html) for details).  The `Dark_Energy_Adaptation` (i.e., NEoST version 3.0.0) is a substantially modified version of NEoST version 2.2.0, as it includes additional functionality to compute neutron stars composed of a MCDF dark energy core and a purely baryonic outer shell surrounding that core. 

NEoST v 3.0.0 - `Dark_Energy_Adaptation` branch
====================================================
This [branch](https://github.com/xpsi-group/neost/tree/Dark_Energy_Adaptation) extends [NEoST v 2.1.0](https://github.com/xpsi-group/neost) with additional functionality for a neutron star with a **dark energy core**, modeled as a MCDF core. This extension works alongside NEoST's existing dense matter and dark matter functionality.

Everything in the base repository of NEoST, i.e., the chiral EFT models, the high-density parameterizations, Multinest nested-sampling algorithm, and the posterior/prior analysis pipeline for purely baryonic stars and ADM admixed stars, are still present. This newest version of NEoST changes the following aspects of the previous version:

- **New TOV solver to handle the MCDF core:** `neost/tovsolvers/TOVde.pyx` (Cython version) and `TOVde_python.py` (Python version) integrate the TOV equations for a neutron star with a dark energy core, with a transition to the purely baryonic shell using the so-called Maxwell construction (please, see the manuscript for further details). Please note that the Python version is set to be the default solver when dark energy is enabled. This is because the Cython solver is currently unstable and only produces identical solutions to the Pythonic version when the integration `step` parameter in `Star.py` is changed from `0.46` to `0.075`. However, upon this change in the `step` parameter, the Cython solver produces no speed up compared to the Python version of the solver, unlike what has been observed with the two-fluid TOV sovlers for dark matter. Thus, while the Cython solver `TOVde.pyx` is included, it should only be used for experimental usage and the Python solver `Tovde_python.py` should be used for accuracy and to reproduce the results found in the manuscript. Lastly, both `TOVde_python.py` and `TOVde.pyx` contain functionality to compute the tidal deformability of a neutron star with an MCDF core. 
- **`Star.py` has two new float arguments and a new boolean argument.** The initiate the `solver_structure()` function in `Star.py`, the new intialization function call is now `Star(epscent, epscent_dm=0.0, eps_plus=0.0, alpha=0.0, enthalpy=False, dark_energy=False)`. Here, the `eps_plus` and `alpha` arguments are the two MCDF parameters describing the transition energy density and the energy density jump parameter (see the manuscript for further details), respectivley. While the boolean argument `dark_energy` toggles whether the TOV solver includes a possible MCDF core. 
- **Updates to `base.py`, `polytropes.py`, `speedofsound.py`, and `tabulated.py`:** More specifically, `base.py` has a new function called `find_max_edscde()`, which computes the maximum central energy density for a star wtih a possible MCDF core, which is the dark energy analogue to the existing `find_max_edsc()`. On the other hand, `polytropes.py`, `speedofsound.py`, and `tabulated.py` now contain a new `adm_type`, called `Dark Energy`, which enables the functionality of computing the neutron star EoS with a dark energy core described by the MCDF EoS. This also grants the user access to the three parameters of the MCDF EoS, namely `alpha` (the energy density jump parameter), `rho_plus` (transition energy density), and `A` (the barotropic fluid paramter). Please see the manuiscript for further details on the specific EoS construction of the MCDF EoS model. 
- **New examples and tutorials:** `Dark_energy_prior.py`, `Dark_energy_posterior.py`, and `DE_MR_Tidal_tutorial.py` demonstrate how to construct the priors, posteriors, and mass–radius/tidal-deformability curves for a star with a possible MCDF core.
- **`Prior.py` and `Likelihood.py` updates.** `Prior.py` samples central density log-uniformly up to `find_max_edsc_de()`'s output when MCDF core configurations are enabled. `Likelihood.py` adds hard cutoffs that exclude non-physical regions of the MCDF parameter space, namely if the sampled central energy density is below the transition energy density, potentially producing twin star solutions, which are disregarded in Rutherford et al. 2026. See the manuscript for further explanation. 
- **`PosteriorAnalysis.py`:** The functions `compute_table_data()` and `compute_auxiliary_data()` in `PosteriorAnalysis.py` take a new boolean argument, `de`, which enables the MCDF core results to be computed when set to `True`.



REPRODUCING PLOTS
=================
All figures in the paper can be reproduced by going to the `plot_routines/` directory and running each python script within the directory. Note, if reproducing the plots via the `plot_routines/` directory, the neost conda enviroment must be activated. Refer to this script and the called plot scripts in `plot_routines/` for a complete account of all options available. The output of the `plot_routines/` directory goes directory to the `plots/` directory. Moreover, some of the scripts in the `plots_routines/` directory pull from the `data/` directory, which contains the posterior samples (and 68% and 95% contour files for plotting) of [J0740](https://arxiv.org/abs/2406.14466), [J0437](https://arxiv.org/abs/2407.06789), and [ST+PDT J0030](https://arxiv.org/abs/2308.09469), which are used in the posterior run scripts and the `Sources_plot.py`.

The most important option is the -r (--repro) flag, which all scripts recognize. By default, the plot routines use the data supplied in the `results/` directory, which contains the results published in the paper, to produce figures. The -r flag tells the plot scripts to instead use user-generated data in the `repro/` directory. 




 RUN SCRIPTS
===========
The run scripts for all prior and posterior inference calculations can be found in the `run_scripts/` directory and can be used to reproduce all of the results in the paper. You need to adapt these scripts if you want to use them with custom run names and/or output directories. These scripts automatically place the results in the `repro/` directory. Furthermore, if you wish to use the plotting scripts with these runs, they must be in the `repro/` directory. The overall structure of the `run_scripts/` folder is as follows:

- `run_scripts/posterior`: All posterior script files

    - `Baryonic_posterior.py` and `Baryonic_prior.py`: The posterior and prior run scripts, which use the [J0740](https://arxiv.org/abs/2406.14466), [J0437](https://arxiv.org/abs/2407.06789), and [J0030](https://arxiv.org/abs/2308.09469) mass-radius posterior inferences. Here, both scripts compute the posterior and prior distributions of the piecewise polytropic (PP) high-density extension EoS model in which all EoS parameters. 
    -  `Bosonic_posterior.py` and `Bosonic_prior.py`: The posterior and prior scripts which consider bosonic ADM admixed neutron stars, described by the [Nelson et al. 2019 bosonic ADM model](https://arxiv.org/abs/1803.03266), using the [J0740](https://arxiv.org/abs/2406.14466), [J0437](https://arxiv.org/abs/2407.06789), and [J0030](https://arxiv.org/abs/2308.09469) mass-radius posterior inferences. These scripts sample both the bosonic ADM and PP EoS parameters, and compare the resulting masses and radii to the above mentioned sources.
    - `Fermionic_posterior.py` and `Fermionic_prior.py`: The posterior and prior scripts which consider fermionic ADM admixed neutron stars, described by the [Nelson et al. 2019 fermionic ADM model](https://arxiv.org/abs/1803.03266), using the [J0740](https://arxiv.org/abs/2406.14466), [J0437](https://arxiv.org/abs/2407.06789), and [J0030](https://arxiv.org/abs/2308.09469) mass-radius posterior inferences. These scripts sample both the fermionic ADM and PP EoS parameters, and compare the resulting masses and radii to the above mentioned sources.
    - `Dark_energy_posterior.py` and `Dark_energy_prior.py`: The posterior and prior scripts which consider neutron stars with an MCDF core, described by the Modified Chaplygin Dark Fluid model used in [Pretel et. al. 2024](https://arxiv.org/abs/2411.08793), using the   [J0740](https://arxiv.org/abs/2406.14466), [J0437](https://arxiv.org/abs/2407.06789), and [J0030](https://arxiv.org/abs/2308.09469) mass-radius posterior inferences. These scripts sample both the MCDF and PP EoS parameters, and compare the resulting masses and radii to the above mentioned sources.
 

OUTPUT FILE STRUCTURES
======================
- `results/`: The first level is either `prior/` or `posterior/`. For both the `prior/` and `posterior/`, there is list of 4 directories: `B/` which constains the baryonic only posterior/prior inferences, `BDM/` which contatins the bosonic ADM admixed posterior/prior inferences, `FDM/` which contains the femrionic ADM admixed posterior/prior inference, and `DE/` which contains the MCDF core and baryonic shell posterior/prior inferences. This is followed by the output files of the runs which vary both the baryonic matter and ADM EoS, namely the output files ending with `post_equal_weights.dat`, `MR_prpr.txt`, `pressures.npy`, `minpres.npy`/`maxpres.npy`, and `minpres_baryon.npy`/`maxpres_baryon.npy`. 
      - `post_equal_weights.dat`: Standard Multinest output file, which containts the EoS model parameters of the corresponding neutron star model being consider, the sampled central density of each source, and log-likelihood evaluation of each sample. Note, the column ordering of the EoS model parameters and the central densities for each source is determined by the ordering of the `variable_params` list in the `run_scripts/` directory.
      - `MR_prpr.txt`: Standard NEoST output file, which contains the corresponding mass and radius samples of each sampled EoS and central density.
      - `pressures.npy`: Standard NEoST output file, which contains the pressures of the considered EoS model.
      - `minpres.npy`/`maxpres.npy`: These are simply the upper (maxpres) and lower (minpres) pressure bounds on the 68% and 95% confidence regions, which are derived from the `pressures.npy` files. The columns go as central energy densities, min/max 68%, and min/max 95%.
      -`minpres_baryon.npy`/`maxpres_baryon.npy`: The same as `minpres.npy`/`maxpres.npy`, but when the dark sector (ADM or MCDF) are neglected in the posterior/prior analysis pipeline, namely in the functions calls within `PosteriorAnalysis.py`. The column structure is the same as that of the `minpres.npy`/`maxpres.npy`. Note, the `minpres_baryon.npy`/`maxpres_baryon.npy` are only relevant for the bosonic ADM, fermionic ADM, and MCDF runs. 

  For the `posterior/` directory, each sub-directory (i.e., `B/`, `BDM/`, `FDM/`, and `DE/`) contains an additional file called `table_data.txt`. The columns of this `table_data.txt file` are the following (with the first column appearing at the top and the last column appearing on the bottom):

- M_TOV
- R_TOV
- log_10(max central energy density)
- max central density as a ratio with respect to n_saturation
- log_10(max central pressure)
- R_1.4
- log_10(central energy density of a 1.4 Msun star), 
- max central density of a 1.4 Msun star as a ratio with respect to n_saturation
- log_10(central pressure of a 1.4 Msun star)
- R_2.0 
- log_10(central energy density of a 2.0 Msun star)
- max central density of a 2.0 Msun star as a ratio with respect to n_saturation
- log_10(central pressure of a 2.0 Msun star)


For the ADM admixed cases, the pressures and energy densities are the ADM pressure/energy density + the PP pressure/energy density. For the cases considering the MCDF cores, the `table_data.txt` only contains M_TOV, R_TOV, R_1.4, and R_2.0 because the manuscript only uses these values.  
