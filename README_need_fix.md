# 260X.XXXX-repro
Reproduction Package for the upcoming paper, titled ``NICER neutron stars with dark energy and dark matter: effects on the inferred equation of state" by Rutherford, Prescod-Weinstein, and Watts 2026 [arXiv preprint to be released on 9/14/26].




NOTICE
======================
Run times range from about day for runs with posteriors which only vary the baryonic matter EoS to about 10-20 days for the priors/posteriors which additionally include the ADM EoS or Modified Chaplygin Dark Fluid (MCDF) EoS on a supercomputing cluster. While it is possible to perform prior and posterior inferences on a personal computer with mpiexec commands (for parallelization), it is NOT recommended, as the computational demand and the deterioration such runs can have on the cpu and other hardware. However, small test runs with about 100 live points can be safely run on a personal computer or laptop to ensure the script is running as desired. 

REQUIREMENTS
============
To reproduce our data and plots, the following additional requirements apply (see also NEoST installation instructions):

  - [neost version 3.0.0](https://github.com/xpsi-group/neost/tree/Dark_Energy_Adaptation). Note, this reproduction package also contains the latest version of the branch as of 9/10/26 for your convenience. See the Dark_Energy_Adaptation directory.
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

Here, Numba is only needed to use the Python TOV solvers. However, normally the much faster Cython TOV solvers should be used (see [https://xpsi-group.github.io/neost/index.html](https://xpsi-group.github.io/neost/index.html) for details).  The Dark_Energy_Adaptation (i.e., NEoST version 3.0.0) is a substantially modified version of NEoST version 2.2.0, as it includes additional functionality to compute neutron stars composed of a MCDF dark energy core and a purely baryonic outer shell surrounding that core. 

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
All figures in the paper can be reproduced by going to the plot_routines/ directory and running each python script within the directory. Note, if reproducing the plots via the plot_routines/ directory, the neost conda enviroment must be activated. Refer to this script and the called plot scripts in plot_routines/ for a complete account of all options available. The output of the plot_routines/ directory goes directory to the plots/ directory. Moreover, some of the scripts in the plots_routines/ directory pull from the data/ directory, which contains the posterior samples of J0740, J0437, and J0030, which are used in the posterior run_scripts.

The most important option is the -r (--repro) flag, which all scripts recognize. By default, generate_figs.sh uses data supplied in the results/ directory---which contains the results published in the paper---to produce figures. The -r flag tells the plot scripts to instead use user-generated data in the repro/ directory. 



STOPPED HERE AT 4:01 PM, NEED TO FINISH TONIGHT!!
 RUN SCRIPTS
========
The run scripts for all prior and posterior inference calculations can be found in the run_scripts/ directory and can be used to reproduce all of the results in the paper. You need to adapt these scripts if you want to use them with custom run names and/or output directories. These scripts automatically place the results in the `repro/` directory. Furthermore, if you wish to use the plotting scripts with these runs, they must be in the `repro/` directory. The overall structure of the run_scripts folder is as follows:

- run_scripts/posterior: All posterior script files
      - /NICER_Real_Data: The real data posterior scripts which use the Riley et al. 2019 & 2021 MR inferences. Here, the NICER_REAL_ADM_VARYING.py script is the script which varies both the Baryonic and ADM equation of state parameters, whereas the NICER_REAL_BARYONIC.py script is the one which neglects the ADM EoS and only considers the Baryonic matter EoS model.
      -  /Future-X: The Future-X data posterior scripts which consider the synthetically generated MR values corresponding to the ADM core model and No ADM models. This directory is further split into the ADM Core Model/ and No ADM Model/ directories. Within the ADM Core Model/ the FUTUREX_ADM_VARYING_BARYONIC.py is the script which samples both the ADM and baryonic matter EoS parameters using the ADM core model sources, while the FUTUREX_ADM_BARYONIC_ONLY.py script is the one which neglected the ADM EoS parameters and only samples the baryonic matter EoS parameters using the ADM core model sources. For the No ADM Model/ directory it is the same as the ADM Core Model/ directory, except now for the No ADM Model sources. Thus the scripts for the including ADM and neglecting ADM scenarios are FUTUREX_NO_ADM_VARYING_BARYONIC.py and FUTUREX_NO_ADM_BARYONIC_ONLY.py, respectively

-  run_scripts/prior: All prior script files
      -  Contains the prior scripts which varies the baryonic matter and ADM EoS parameters named FERMIONIC_PRIOR.py.
      -  /Appendix B: Contains the run scripts to generate the data used in Appendix B, which is the approximation of zero self-repulsion using 10^{-5} MeV^{-1}.
 

OUTPUT FILE STRUCTURES
======================
- results/: The first level is either "prior" or "posterior". For the prior, this is followed by the output files of the runs which vary both the baryonic matter and ADM EoS, namely the output files ending with post_equal_weights.dat, MR_prpr.txt, and pressures.npy.
      - post_equal_weights.dat: Standard Multinest output file, which containts the admixed (baryonic + ADM) EoS model parameters of the PP model and fermionic ADM EoS, the sampled central density, and log-likelihood evaluation of each sample.
      - MR_prpr.txt: Standard NEoST output file, which contains the corresponding mass and radius samples of each sampled admixed EoS and central density.
      - pressures.npy: Standard NEoST output file, which contains the pressures of the baryonic EoS in which ADM was considered during prior sampling.
  Moreover, the prior directory is also followed by two other directories:
      - Appendix_B/: Relative radial percent differences comparing zero ADM self-repulsion to 10^{-5} MeV^{-1}. The tail end of the file indicates which choice of baryonic EoS is used and if a different step of ADM particle mass or mass-fraction were used. For example, the file Relcent_diff_intermediate_stiff_baryonic_newfchistep.npy are the results which used the intermediately stiff baryonic EoS in the manuscript with a smaller step in fchi compared to the Relcent_diff_intermediate_stiff_baryonic.npy file.
       - Fchi_prior_calculation/: Directory containing the scripts that print the estimated ADM mass-fraction and distances to the Galactic center for J0437 and J0030. Note, these are run scripts because the results are simply printed results and not stored in a file structure.

  For the posterior, the directory is followed by two more directories, Future-X/ and NICER_Real_Data/, which are the posteriors corresponding to the Future-X and Real data inferences in the manuscript, respectively.
    - Future-X/: Followed by two more directories for the ADM_core_model/ and No_ADM_Model/, which of course correspond to the ADM core model and No ADM model, respectively. Both of these directories then have the output directories for the posteriors in which ADM is included (FUTUREX_ADM_VARYING_BARYONIC for the ADM core model and FUTUREX_NO_ADM_VARYING_BARYONIC for the No ADM model) and neglected from sampling (FUTUREX_ADM_BARYONIC_ONLY for the ADM core model and FUTUREX_NO_ADM_BARYONIC_ONLY for the No ADM model). These output directories have the post_equal_weights.dat, MR_prpr.txt, and pressures.npy files. Furthermore, they also have the minpres.npy and maxpres.npy file, which are simply the upper (maxpres) and lower (minpres) pressure bounds on the 68% and 95% confidence regions, which are derived from the pressures.npy files.

    - NICER_Real_Data/: Followed by the output directories NICER_REAL_ADM_VARYING_BARYONIC and NICER_REAL_BARYONIC, which are the directories which include ADM and neglect it during sampling, respectively. The output files contain the same output files as that of the Future-X/ posteriors. 
