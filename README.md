# GARPOS-MCMC

"GARPOS-MCMC" (GNSS-Acoustic Ranging combined POsitioning Solver with MCMC) is an analysis tool for GNSS-Acoustic seafloor positioning.

### Version
Latest version is GARPOS v1.0.0 (Apr. 14. 2022)

#### Major change(s)
* v1.0.0: first release

# Citation

### for methodology

Watanabe, S., Ishikawa, T., Nakamura, Y., & Yokota, Y. (2022). Title, EarthArXiv. (https://doi.org/ppp/sss).

### for code
Shun-ichi Watanabe, Tadashi Ishikawa, Yuto Nakamura & Yusuke Yokota. (2022). GARPOS-MCMC (Version 1.0.0). Zenodo. (DOI will be available soon)

## Corresponding author

* Shun-ichi Watanabe
* Hydrographic and Oceanographic Department, Japan Coast Guard
* Website : https://www1.kaiho.mlit.go.jp/KOHO/chikaku/kaitei/sgs/index.html (in Japanese)


# License

"GARPOS-MCMC" is distributed under the [GPL 3.0] (https://www.gnu.org/licenses/gpl-3.0.html) license.


## Algorithm and documentation

Please see Watanabe, S., Ishikawa, T., Nakamura, Y., & Yokota, Y. (2022). Title, EarthArXiv. (https://doi.org/ppp/sss).

### Models for perturbation field

For the detail, the users should read the above paper.

#### Double-grad (mode=m100)

Independently esitmate the spatial gradient parameters related to the sea-surface and seafloor instruments' positions. 
This is the similar condition to the conventional GARPOS (https://github.com/s-watanabe-jhod/garpos).

#### Single-grad (mode=m101)

Constrain the directions of the spatial gradient parameters related to the sea-surface and seafloor instruments' positions, by estimating a propotinal constant for the seafloor gradient to the sea-surface gradient (between the range [0, 1]).

#### Alpha2-offset (mode=m102)

In addition to the "single-grad", a constant offset vector for the seafloor gradient is estimated. 

### Note

This is an enhanced version of conventional "GARPOS" (https://github.com/s-watanabe-jhod/garpos) based on empirical Bayes approach. Many variables, data format and usages are identical to the conventional GARPOS.

For the conventional GARPOS methodology, please see Watanabe, S., Ishikawa, T., Yokota, Y., and Nakamura, Y., (2020) https://doi.org/10.3389/feart.2020.597532

# Requirements

* Python 3.7.3
* Packages NumPy, Scipy, Pandas, Matplotlib, and Scikit-sparse are also required.
* Fortran 90 compiler (e.g., gfortran)

Environments under [Anaconda for Linux](https://www.anaconda.com/distribution/) is tested.


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

"bin/GARPOS_mcmc_v1.0.0.py" is a driver code. 
An observation dataset is stored in "sample" directory as demo data.

NOTE: Unlike conventional GARPOS, travel-time outliers must be removed before the MCMC run.

To solve position with array-constraint condition (for epoch TOS2.1803.meiyo_m4),

```bash
cd sample
# for double-grad model (mode:m100)
../bin/GARPOS_mcmc_v1.0.0.py -i Settings-mcmc-demo.ini -f cfgfix/TOS2/TOS2.1803.meiyo_m4-fix.ini -d demo100/TOS2 --mode m100
# for single-grad model (mode:m101)
../bin/GARPOS_mcmc_v1.0.0.py -i Settings-mcmc-demo.ini -f cfgfix/TOS2/TOS2.1803.meiyo_m4-fix.ini -d demo101/TOS2 --mode m101
# for alpha2-offset model (mode:m102)
../bin/GARPOS_mcmc_v1.0.0.py -i Settings-mcmc-demo.ini -f cfgfix/TOS2/TOS2.1803.meiyo_m4-fix.ini -d demo102/TOS2 --mode m102
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

+ GARPOS_mcmc_v1.0.0.py
  + init_position (in setup_model.py)
  + make_knots (in setup_model.py)
  + setup_hparam (in setup_model.py)
  + atd2enu (in forward.py)
    + corr_attitude (in coordinate_trans.py)
      + llh2xyz (in coordinate_trans.py)
      + xyz2enu (in coordinate_trans.py)
  + data_var_base (in setup_model.py)
  + jacobian (in func_*.py)
    + calc_gamma (in forward.py)
  + hparam_to_real (in eachstep.py)
  + calc_traveltime (in traveltime.py)
  + E_matrix (in eachstep.py)
  + H_matrix (in func_*.py)
    + derivative2 (in eachstep.py)
  + log_likelihood (in eachstep.py)
  + sampling_a (in eachstep.py)
  + a_to_mp (in func_*.py)
  + calc_gamma (in forward.py)
  + calc_forward (in forward.py)
    + corr_attitude (in coordinate_trans.py)
      + llh2xyz (in coordinate_trans.py)
      + xyz2enu (in coordinate_trans.py)
    + calc_traveltime (in traveltime.py)
  + outresults (in output.py)
    + write_cfg (in output.py)
  + plot_residuals (in resplot.py)
