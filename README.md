# CausticSNUpy
Python module for drawing caustic lines and determining cluster membership of galaxies using the caustic technique presented in Diaferio (1999) and Serra et al. (2011).  

CausticApp, a program written by Serra and Diaferio, was used as a benchmark for this module and thus we tried to reproduce the same output for a given input.

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
This package is currently only available through manual installation.  
After downloading and the tar.gz file, follow one of the following methods.

### Method 1: using pip
1. Open terminal and change directory to where tar.gz file is located.
2. Type and run `pip install` + tar.gz file name.
```
pip install CausticSNUpy-0.0.1.tar.gz
```

### Method 2: run setup.py
1. Open terminal and change directory to where setup.py is located.
2. In your terminal, type and run
```
python setup.py install
```
3. Now you are ready to use this package!

## How to uninstall
1. Open terminal (any directory is fine).
2. In your terminal, type and run 
```
pip uninstall CausticSNUpy
```

## How to use
### Input file
Input file must follow the following format:
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
* `r` : projected distance from the cluster center to each galaxy
* `v` : relative (with regard to the cluster center) l.o.s. velocity of each galaxy
* `r_grid` : grid along the r-axis of the redshift diagram
* `A` : amplitude of the caustic lines, evaluated for r_grid

## References
1. Diaferio, A. 1999, MNRAS, 309, 610
2. Gifford, D., Miller, C., & Kern, N. 2013, ApJ, 773, 116
3. Serra, A. L., Diaferio, A., Murante, G., & Borgani, S. 2011, MNRAS, 412, 800
4. Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London