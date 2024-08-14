# GARPOS-MCMC

<img src="https://github.com/s-watanabe-jhod/garpos/assets/68180987/ed955d3c-4c3b-4ca3-91d5-f57b876cfa7b" width=400 alt="GARPOS">

GARPOS-MCMC (GNSS-Acoustic Ranging combined POsitioning Solver with MCMC) is an analysis tool for GNSS-Acoustic seafloor positioning.

### Version
Latest version is GARPOS v2.0.0 (Aug. 14. 2024)

* Version 2 has the function for model selection with WBIC (widely applicable Bayesian Information Criterion).

#### Major change(s)
* v2.0.0: Implement the model selection scheme with WBIC.
* v1.2.0: Transformation methods for estimation parameters are changed. The format for configuration file is changed. 
* v1.1.0: Skip raytrace in each MCMC step with the pre-calculated travel times.
* v1.0.0: first release

# Citation

### for methodology (version 2 with WBIC)

Watanabe, S., Ishikawa, T., Nakamura, Y., & Yokota, Y. (in prep.). Model selection for the sound speed perturbation of the GNSS-A using the widely applicable Bayesian Information Criterion (WBIC). preprint will be available soon.

### for methodology (version 1 or only for MCMC)

Watanabe, S., Ishikawa, T., Nakamura, Y., & Yokota, Y. (2023). Full-Bayes GNSS-A solutions for precise seafloor positioning with single uniform sound speed gradient layer assumption. J. Geod. 97, 89. https://doi.org/10.1007/s00190-023-01774-6

### for code

Shun-ichi Watanabe, Tadashi Ishikawa, Yuto Nakamura & Yusuke Yokota. (2024). GARPOS-MCMC: MCMC-based analysis tool for GNSS-Acoustic seafloor positioning (v2.0.0) Zenodo. 

## Corresponding author

* Shun-ichi Watanabe
* Hydrographic and Oceanographic Department, Japan Coast Guard
* Website : https://www1.kaiho.mlit.go.jp/KOHO/chikaku/kaitei/sgs/index.html (in Japanese)


# License

"GARPOS-MCMC" is distributed under the [GPL 3.0] (https://www.gnu.org/licenses/gpl-3.0.html) license.


## Algorithm and documentation

Please see the literature in "citation for methodology".

### Models for perturbation field

For the detail, the users should read the above papers. This version implements m100/m101/m102 models as candidates.

#### model m100

Independently esitmate the spatial gradient parameters related to the sea-surface and seafloor instruments' positions. 
This is the similar condition to the conventional GARPOS (https://github.com/s-watanabe-jhod/garpos).

#### model m101

Constrain the directions of the spatial gradient parameters related to the sea-surface and seafloor instruments' positions, by estimating a propotinal constant for the seafloor gradient to the sea-surface gradient (between the range [0, 1]).

#### model m102

In addition to the model "m101", a constant offset vector for the seafloor gradient is estimated. 

### Note

This is an enhanced version of conventional "GARPOS" (https://github.com/s-watanabe-jhod/garpos) based on empirical Bayes approach. Many variables, data format and usages are identical to the conventional GARPOS.

For the conventional GARPOS methodology, please see Watanabe, S., Ishikawa, T., Yokota, Y., and Nakamura, Y., (2020) https://doi.org/10.3389/feart.2020.597532

# Requirements

Environments under [Anaconda for Linux] (https://www.anaconda.com/distribution/) is tested.

* Python 3.11.7 is tested.
* Packages tqdm, NumPy, Scipy, Pandas, Matplotlib, and Scikit-sparse are also required.
  * NOTE: some reported that "sksparse" cannot be used on Apple M1 Chip. 
* Fortran 90 compiler (e.g., gfortran)

### Compilation of Fortran90-based library

For the calculation of travel time, a Fortran90-based library is needed.
For example, the library can be compiled via gfortran as,

```bash
gfortran -shared -fPIC -fopenmp -O3 -o lib_raytrace.so sub_raytrace.f90 lib_raytrace.f90
```

Path to the library should be indicated in "Settings.ini".


# Usage

When using GARPOS-MCMC, you should prepare the following files.
* Initial site-parameter file (e.g., *initcfg.ini)
* Acoustic observation data csv file
* Reference sound speed data csv file
* Settings file (e.g., Settings.ini)

"bin/solveSingleEpoch_mcmcv200.py" is a driver code. 
An observation dataset is stored in "sample" directory as demo data.

NOTE: Unlike conventional GARPOS, travel-time outliers must be removed before the MCMC run.

To solve position with array-constraint condition (for epoch TOS2.1803.meiyo_m4) for WBIC-preferred model,

```bash
cd sample
../bin/solveSingleEpoch_mcmcv200.py -i Settings-mcmc-demo.yml -f cfgfix/TOS2/TOS2.1803.meiyo_m4-fix.ini -d demo/TOS2
```

The following files will be created in the directory (specified with "-d" option).

* MCMC results
  * MCMC sample chain and plot (*chain.csv, *chain.pdf)
  * Plot of MCMC samples' histogram (*hist.pdf)
  * Statistics for MCMC chain (*percentile.csv)
* Conventional GARPOS outputs (for the maximum likelihood MCMC sample)
  * Estimated site-parameter file (*res.dat)
  * Modified acoustic observation data csv file (*obs.csv)
  * Model parameter list file (*m.p.dat)
  * A posteriori covariance matrix file (*var.dat)
  * Travel-time residual plot (fig/*t.s.png)


# Note

Please be aware of your memory because it stores all MCMC samples for test.


### List of functions

+ solveSingleEpoch_mcmcv120.py
  + garpos_mcmc.py
    + setup_hparam (in setup_model.py)
    + init_position (in setup_model.py)
    + atd2enu (in forward.py)
      + corr_attitude (in coordinate_trans.py)
        + llh2xyz (in coordinate_trans.py)
        + xyz2enu (in coordinate_trans.py)
    + calc_snell (in traveltime_d.py)
    + make_knots (in setup_model.py)
    + derivative2 (in setup_model.py)
    + H_bases (in func_*.py)
    + jacobian (in func_*.py)
      + calc_gamma (in forward.py)
    + data_var_base (in setup_model.py)
    + hparam_to_real (in eachstep.py)
    + calc_traveltime (in traveltime_d.py)
    + calc_traveltime_raytrace (in traveltime_rt.py)
    + E_matrix (in eachstep.py)
    + H_params (in func_*.py)
    + H_matrix (in func_*.py)
    + sampling_a (in eachstep.py)
    + a_to_mp (in func_*.py)
    + calc_gamma (in forward.py)
    + calc_forward (in forward.py)
      + corr_attitude (in coordinate_trans.py)
        + llh2xyz (in coordinate_trans.py)
        + xyz2enu (in coordinate_trans.py)
      + calc_traveltime_raytrace (in traveltime_rt.py)
    + outresults (in output.py)
      + write_cfg (in output.py)
    + plot_residuals (in resplot.py)
  
