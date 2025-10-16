# CausticSNUpy
Python implementation of the caustic technique for identifying galaxy cluster membership and measuring the mass profile of the cluster. Original technique is presented in [Diaferio (1999)](https://ui.adsabs.harvard.edu/abs/1999MNRAS.309..610D/abstract) and [Serra et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.412..800S/abstract).  

CausticApp, a program written by Serra and Diaferio, was used as a benchmark for this module and thus we tried to reproduce the same output for a given input.

Details of this code can be found in [Kang et al. (2024)](http://doi.org/10.3847/1538-4365/ad390d).

## Citation
We kindly request you to cite the following paper if you used this code:

[Kang, W., Hwang, H. S., Song, H., et al. 2024](http://doi.org/10.3847/1538-4365/ad390d)

## Dependencies
### Required Packages
* NumPy
* SciPy
* astropy
* scikit-image

### Tested with:
* Python 3.9.7
* numpy 1.20.3
* scipy 1.7.1
* astropy 4.3.1
* scikit-image 0.18.3

## How to install
Installatation is available through `pip` or direct install from the githup repo.

### Method 1: installing from pip repository
1. On your terminal, type and enter `pip install causticsnupy`.
```
pip install CausticSNUpy
```

### Method 2: installing from zip file
1. Download the `.zip` file from [github](https://github.com/woodykang/CausticSNUpy) repository. Click 'Code' button on the top right of the page, and click 'Download ZIP'.
2. In your terminal, change directory to the location where the `.zip` file is.
3. Type and enter the following:
```
pip install CausticSNUpy-main.zip
```

### Method 3: run setup.py
1. Clone the repository to your local machine.
2. Open terminal and change directory to where `setup.py` is located.
3. In your terminal, type and run
```
python setup.py install
```

## How to uninstall
1. Open terminal (any directory is fine).
2. In your terminal, type and run 
```
pip uninstall CausticSNUpy
```

## How to use
### Input file
If you are using `run_from_file`, the input file must follow the following format:
* The first line of the input file may either have 1 or 4 numbers.
    * The first line should be in format `N` or `N RA DEC VEL`,
    where N is the number of galaxies in the data, RA and DEC are the right ascension and declination of the cluster center (both in deg), and VEL is the radial velocity of the cluster center (in km/s).
    * If RA, DEC, VEL are not given, input parameter `center_given` should be set to `False`.
    * If N does not match the number of galaxies listed, the program will emit and error.
* Input file must have 3 columns (except for the first line, as mentioned above).
    * Each line is in format `RA DEC VEL`. RA and DEC are the right ascension and declination of each galaxy (both in deg), and VEL is the radial velocity of each galaxy (in km/s).

### Routine
See `example.ipynb` for the code to set the parameters, run the caustic method, and plot the redshift diagram with caustic lines.

### Main attributes of the class
* `r` : projected distance from the cluster center to each galaxy [Mpc]
* `v` : relative (with regard to the cluster center) l.o.s. velocity of each galaxy [km/s]
* `r_grid` : grid along the r-axis of the redshift diagram [Mpc]
* `A` : amplitude of the caustic lines along `r_grid` [km/s]
* `dA` : uncertainty in `A` [km/s]
* `M` : enclosed mass profile along `r_grid` [$M_{\odot}$]
* `dM` : uncertainty in `M` [$M_{\odot}$]

## References
1. Diaferio, A. 1999, MNRAS, 309, 610
2. Gifford, D., Miller, C., & Kern, N. 2013, ApJ, 773, 116
3. Serra, A. L., Diaferio, A., Murante, G., & Borgani, S. 2011, MNRAS, 412, 800
4. Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London