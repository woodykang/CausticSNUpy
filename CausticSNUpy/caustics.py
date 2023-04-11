# This class is for applying caustic method as presented by Diaferio 1999 and Serra et al. 2011.
# When a list of galaxy data containing RA, Dec, and l.o.s. velocity is given, this class calculates the caustic line and determines the membership of each galaxy.

# Code written by Woo Seok Kang, Astronomy Program, Dept. of Physics and Astronomy, Seoul National University (SNU)
# with help from members of Exgalcos team, SNU.
# If you have any questions or comments, contact woodykang@snu.ac.kr

# References:
# 1. Diaferio, A. 1999, MNRAS, 309, 610
# 2. Gifford, D., Miller, C., & Kern, N. 2013, ApJ, 773, 116
# 3. Serra, A. L., Diaferio, A., Murante, G., & Borgani, S. 2011, MNRAS, 412, 800
# 4. Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London

import numpy as np
import astropy.coordinates
import astropy.units
import astropy.cosmology
import skimage.measure
from .hierarchical_clustering import hier_clustering

class Caustics:

    '''
    Python implementation for caustic technique presented in Diaferio 1999 and Serra et al. 2011.

    Parameters
    ---------------------------
    fpath           : str,      file path of data
    v_lower         : float,    lower limit of l.o.s. velocity that member galaxies should have (in units of km/s) 
    v_upper         : float,    upper limit of l.o.s. velocity that member galaxies should have (in units of km/s)
    v_max           : float,    maximum l.o.s. velocity from the cluster center when drawing the redshift diagram (in units of km/s)
    r_max           : float,    maximum projected distance that member galaxies should have (in units of Mpc)
    center_given    : bool,     whether RA, Dec, l.o.s. velocity is given for the cluster center at the first line of the data file; default value False
    H0              : float,    current value of Hubble parameter (in units of km/s); default value 100
    Om0             : float,    matter density parameter for the cosmological model; default value 0.3
    Ode0            : float,    dark energy density paramter for the cosmological model; default value 0.7
    Tcmb0           : float,    temperature of the CMB at z=0 (in units of Kelvin); default value 2.7
    q               : float,    scaling factor (see Section 4.1 of Diaferio 1999); default value 25
    r_res           : int,      resolution of the redshift diagram in the r-axis; default value 100
    v_res           : int,      resolution of the redshift diagram in the v-axis; default value 100
    BT_thr          : str,      Binary Tree treshold, should be either "AD" or "ALS" (currently only ALS is supported); default value "ALS"
    gal_m           : float,    mass of single galaxy assumed for hierarchical clustering method; default value 1e12
    h_c             : float,    smoothing length used for KDE; if None, the program finds it; default value None
    kappa           : float,    user's choice of kappa for caustic determination; if None, the program finds it; if an input is given, kappa is fixed; default value None
    alpha           : float,    coefficient for smoothing factor h_c; this is multiplied to the final value of h_c if the user thinks the caustic is too noisy; default value 1
    grad_limit      : float,    maximum value of d ln(A) / d ln(r) permitted (see Section 4.3, Serra et al. 2011); default value 2 
    display_log     : bool,     prints the log; default value True

    Attributes
    ---------------------------
    run                 : function,         executes the procedure; corresponds to the main program
    create_member_list  : function,         creates a txt file with column indicating membership (1 for member and 0 for interloper) appended to the original data given as input
    r_grid              : numpy ndarray,    grid value along the r-axis of the redshift diagram (in units of Mpc)
    v_grid              : numpy ndarray,    grid value along the v-axis of the redshift diagram (in units of km/s)
    A                   : numpy ndarray,    amplitude of the caustic line along r_grid (in units of km/s)
    r                   : numpy ndarray,    projected distance from cluster center to each galaxy within cutoff limits (in units of Mpc)
    v                   : numpy ndarray,    relative l.o.s. velocity of each galaxy within cutoff limits (in units of km/s)
    member              : numpy ndarray,    bool array which indicates whether a galaxy is a member or not; this is only applied to galaxies after projected distance and velocity cutoffs are applied (i.e., for attributes r and v); if you want to determine the membership of the total galaxies listed in the input file, please use `create_member_list`.
    den                 : numpy ndarray,    number density of galaxies on the redshift diagram calculated for r_grid and v_grid
    kappa               : float,            level at which the caustic line is drawn
    cl_ra               : float,            RA of cluster center (in units of deg); if center_given == True, then it is same as the value given in the input file
    cl_dec              : float,            Dec of cluster center (in units of deg); if center_given == True, then it is same as the value given in the input file
    cl_v                : float,            radial velocity of cluster center (in units of km/s); if center_given == True, it is same as the value given in the input file
    R                   : float,            mean projected distance from the cluster center to candidate member galaxies (in units of Mpc)
    vvar                : float,            variance of relative l.o.s. velocity of candidate member galaxies (in units of (km/s)^2)

    Note
    ---------------------------
    The input file containing the observed data must abide the following form:
        The first row must be the `N`, where N is the number of galaxies listed in the file or
        `N RA Dec v`, where N is the number of galaxies listed in the file, RA is the right ascension of the cluster center (in units of deg), Dec is the declination of the cluster center (in units of deg), and v is the radial velocity of the cluster center (in units of km/s)
        
        From the second line until N+1 line, the first column must be the RA of each galaxy (in units of deg), the second column must be the Dec of each galaxy (in units of deg), and the third column is the l.o.s. velocity of each galaxy (in units of km/s).
    '''


    c = 299792.458                              # speed of light in km/s

    def __init__(self, fpath, v_lower, v_upper, v_max, r_max, center_given=False, H0=100, Om0=0.3, Ode0=0.7, Tcmb0=2.7, q=25, r_res=100, v_res=100, BT_thr="ALS", gal_m=1e12, h_c = None, kappa=None, alpha=1, grad_limit=2, display_log=True):
        self.fpath = fpath                      # str,      file path of data
        self.v_lower = v_lower                  # float,    lower limit of l.o.s. velocity that member galaxies should have (in units of km/s)
        self.v_upper = v_upper                  # float,    upper limit of l.o.s. velocity that member galaxies should have (in units of km/s)
        self.v_max = v_max                      # float,    maximum l.o.s. velocity from the cluster center when drawing the redshift diagram (in units of km/s)
        self.v_min = -v_max                     # float,    minimum l.o.s. velocity from the cluster center when drawing the redshift diagram (in units of km/s)
        self.r_max = r_max                      # float,    maximum projected distance that member galaxies should have (in units of Mpc)
        self.center_given = center_given        # bool,     whether RA, Dec, l.o.s. velocity is given for the cluster center at the first line of the data file; default value False
        self.H0 = H0                            # float,    current value of Hubble parameter (in units of km/s); default value 100
        self.Om0 = Om0                          # float,    matter density parameter for the cosmological model; default value 0.3
        self.Ode0 = Ode0                        # float,    dark energy density paramter for the cosmological model; default value 0.7
        self.Tcmb0 = Tcmb0                      # float,    temperature of the CMB at z=0; default value 2.7
        self.q = q                              # float,    scaling factor (see Section 4.1 of Diaferio 1999); default value 25
        self.r_res = r_res                      # int,      resolution of the redshift diagram in the r-axis; default value 100
        self.v_res = v_res                      # int,      resolution of the redshift diagram in the v-axis; default value 100
        self.BT_thr = BT_thr                    # str,      Binary Tree treshold, should be either "AD" or "ALS" (currently only ALS is supported); default value "ALS"
        self.gal_m = gal_m                      # float,    mass of single galaxy assumed for hierarchical clustering method; default value 1e12
        self.h_c = h_c                          # float,    smoothing length used for KDE; if None, the program finds it; default value None
        self.kappa = kappa                      # float,    user's choice of kappa for caustic determination; if None, the program finds it; if an input is given, kappa is fixed; default value None
        self.alpha = alpha                      # float,    coefficient for smoothing factor h_c; this is multiplied to the final value of h_c if the user thinks the caustic is too noisy; default value 1
        self.grad_limit = grad_limit            # float,    maximum value of d ln(A) / d ln(r) permitted (see Section 4.3, Serra et al. 2011); default value 2
        self.display_log = display_log          # bool,     prints the log; default value True
        
        # Declare other class attributes
        self.BT_mainbranch = None
        self.BT_sigma = None
        self.BT_cut_idx = None
        self.cand_mem_idx = None
        
        self.r_min = 0                          # minimum of r should be, obviously, 0
        self.r_grid = None
        self.v_grid = None
        self.contours = None
        self.N = None
        self.cl_ra = None
        self.cl_dec = None
        self.cl_v = None
        self.d_A = None
        self.gal_ra = None
        self.gal_dec = None
        self.gal_v = None
        
        self.R = None
        self.vvar = None
        
        self.r = None
        self.v = None
        self.r_cutoff_idx = None
        self.v_cutoff_idx = None
        self.rv_mask = None
        
        self.den = None
        self.A = None
        self.member = None
        self.full_member = None
        

    def run(self):
        self.unpack_data()                                                  # unpack data from given inputs

        x_data = self.r*self.H0                                             # rescaled data points in r-axis (in units of km/s)
        y_data = self.v/self.q                                              # rescaled data points in v-axis (in units of km/s)

        r_grid = np.linspace( self.r_min, self.r_max, self.r_res)            # grid of r-axis on the redshift diagram
        v_grid = np.linspace(-self.v_max, self.v_max, self.v_res)            # grid of v-axis on the redshift diagram

        x_grid = r_grid*self.H0                                             # grid of rescaled r-axis (in units of km/s)
        y_grid = v_grid/self.q                                              # grid of rescaled v-axis (in units of km/s)

        if self.display_log == True:
            print("Estimating number density.")
        f = self.density_estimation(x_data, y_data)                         # function that returns the number density on the redshift diagram when coordianates (x, y) are given
        if self.display_log == True:
            print("Number density estimation done.")
            print("")

        X, Y = np.meshgrid(x_grid, y_grid)                                  # mesh grid
        den = f(X, Y)*self.H0/self.q*2                                      # number density estimated at each point X, Y, normalized to be 1 when integrated along r, v axis

        a = den.min()
        b = den.max()
        fn = lambda kappa: self.S(kappa, r_grid, v_grid, self.r_res, den, self.R, self.vvar)            # lambda function is used to fix other parameters except for kappa; thus, fn is only a function of kappa and is the S(k) function described in Diaferio 1999
        if self.kappa is None:
            # minimize S(k) to get optimal kappa value
            if self.display_log == True:
                print("Minimizing S(k) to find kappa.")
            self.kappa = self.find_kappa(r_grid, v_grid, den)
            if self.display_log == True:
                print("kappa found. kappa =  {:.5e}, S(k) = {:.5e}.".format(self.kappa, fn(self.kappa)))
                print("")
        else:
            if self.display_log == True:
                print("User input for kappa = {:.5e}, S(k) = {:.5e}".format(self.kappa, fn(self.kappa)))
    
        # calculate A(r) with the minimized kappa
        if self.display_log == True:
            print("Drawing caustic lines.")
        A = self.calculate_A(self.kappa, den, r_grid, v_grid)                    # calculate the final amplitude of caustic lines
        self.contours = skimage.measure.find_contours(den, self.kappa)
        if self.display_log == True:
            print("Caustic line calculation done.\n")

        # determine membership
        if self.display_log == True:
            print("Determining membership.")
        member = self.membership(self.r, self.v, r_grid, A)                 # numpy array where the values are 1 for members and 0 for interlopers; this is only applied to members within the cutoff limits
        full_member = np.zeros(self.N).astype(bool)
        full_member[self.rv_mask] = member
        if self.display_log == True:
            print("Membership determination done.\n")

        # set variables to class attributes
        self.r_grid = r_grid
        self.v_grid = v_grid
        self.den = den
        self.A = A
        self.member = member
        self.full_member = full_member


    def create_member_list(self, new_fpath = None):

        '''
        Create a txt file with a fourth column appended to the original galaxy data indicating the membership.
        If directory of the new file (new_path) is unspecified, the new file will be saved at fpath + ".member.txt", where fpath is the file path to the data.
        If new_path is given, the file will be saved as new_path.
        '''

        if new_fpath is None:
            new_fpath = self.fpath + ".member.txt"

        input_f = open(self.fpath, 'r')
        output_f = open(new_fpath, 'w')
        lines = input_f.readlines()
        output_f.write(lines[0])
        for i in range(len(self.full_member)):
            line = lines[i+1].rstrip()
            output_f.write(line + "{:5d}".format(self.full_member[i]) + '\n')
        input_f.close()
        output_f.close()

    def unpack_data(self):

        '''
        Unpack data from input file.
        Apply velocity and projected distance cutoff given as inputs.
        Construct binary tree from hierarchical clustering and find candidate members.
        Calculate projected distance from cluster center (r) and relative l.o.s. velocity (v).
        '''
        
        if self.display_log == True:
            print("Unpacking data.")
    
        gal_ra, gal_dec, gal_v = np.loadtxt(self.fpath, skiprows=1, unpack=True)        # RA (deg), Dec (deg), l.o.s velocity (km/s)
        cluster_data = np.loadtxt(self.fpath, max_rows = 1)                             # In the first row, the format is N, cl_ra, cl_dec, cl_v
                                                                                        # where N is number of galaxies observed, and
                                                                                        # cl_ra, cl_dec, cl_z are RA (deg), DEC (deg), los velocity (km/s) of the cluster (if specified)
        
        if self.center_given == True:
            if np.size(cluster_data) == 1:
                raise Exception("First row of input file has only one number, while the center coordianates should be given.")
            N = int(cluster_data[0])                                                    # number of galaxies given as input
            
        else:
            if np.size(cluster_data) == 1:
                N = int(cluster_data)
            else:
                N = int(cluster_data[0])
        
        if N != gal_ra.size:                                                            # emit error if N does not match the actual number of galaxies listed in the file
            raise Exception("Number of galaxies stated in the first line of file does not match the number of galaxies listed.")
        
        # apply velocity cutoff
        v_cutoff_idx = (gal_v > self.v_lower) & (gal_v < self.v_upper)                  # numpy array where values are 1 for galaxies within the l.o.s. velocity limit and 0 for galaxies outside the l.o.s. velocity limit

        # shortlist candidate members using hierarchical clustering
        cand_mem_idx, mainbranch, BT_sigma, BT_cut_idx = hier_clustering(gal_ra, gal_dec, gal_v, mask=v_cutoff_idx)             # indices of candidate members, calculated from hierarchical clustering; see Appendix A, Diaferio 1999 and Section 4, Serra et al. 2011
        
        if self.display_log == True:
            print("Hierarchical clustering done.")
        if self.display_log == True:
            print("Number of candidate members : {}".format(sum(cand_mem_idx)))

        if not self.center_given:                                                                       # if coordinates of the cluster center is not given by the user, 
            cl_ra, cl_dec, cl_v = self.find_cluster_center(gal_ra, gal_dec, gal_v, cand_mem_idx)        # calculate it using the candidate members; see Section 4.3, Serra et al. 2011
            if self.display_log == True:
                print("Cluster center : RA = {:4.5f} deg, Dec =  {:4.5f} deg, v = {:6f} km/s".format(cl_ra, cl_dec, cl_v))
        else:
            cl_ra, cl_dec, cl_v = cluster_data[1:]
        
        # calculate projected distance and radial velocity
        LCDM = astropy.cosmology.LambdaCDM(self.H0, self.Om0, self.Ode0, self.Tcmb0)                    # Lambda CDM model with the given parameters
        
        cl_z = cl_v/self.c                                                                              # redshift of the cluster center
        d_A = LCDM.angular_diameter_distance(cl_z)                                                      # angular diameter distance to the cluster center
        
        angle = astropy.coordinates.angular_separation(cl_ra*np.pi/180, cl_dec*np.pi/180, gal_ra*np.pi/180, gal_dec*np.pi/180)      # angular separation of each galaxy and cluster center
        r = (np.sin(angle)*d_A).to(astropy.units.Mpc).value                                                                         # projected distance from cluster center to each galaxy (in Mpc)
        v = (gal_v - cl_v)/(1+cl_z)                                                                                                 # relative l.o.s velocity with regard to cluster center

        R = np.average(r[cand_mem_idx])             # average projected distance from the center of the cluster to candidate member galaxies (in units of Mpc); later to be used for function S(k)
        vvar = np.var(v[(cand_mem_idx) & (r < R)])      # variance of v calculated from candidate members (in units of (km/s)**2); later to be used for function S(k)

        if self.display_log == True:
            print("Mean distance       : {:4.5f} Mpc".format(R))
            print("Velocity Dispersion : {:4.5f} km/s".format(np.std(v[cand_mem_idx])))

        # apply projected distance cutoff
        r_cutoff_idx = (r < self.r_max)                                                 # numpy array where values are 1 for galaxies within r_max and 0 for galaxies outside r_max

        r = r[v_cutoff_idx & r_cutoff_idx]                                              # apply cutoff to projected distance from the cluster center to each galaxy
        v = v[v_cutoff_idx & r_cutoff_idx]                                              # apply cutoff to relative l.o.s. velocity
        
        # set local varibales to class attribute
        self.N = N
        self.cl_ra = cl_ra
        self.cl_dec = cl_dec
        self.cl_v = cl_v
        self.d_A = d_A

        self.gal_ra = gal_ra
        self.gal_dec = gal_dec
        self.gal_v = gal_v

        self.r = r
        self.v = v

        self.vvar = vvar
        self.R = R

        self.v_cutoff_idx = v_cutoff_idx
        self.cand_mem_idx = cand_mem_idx
        self.r_cutoff_idx = r_cutoff_idx
        self.rv_mask = v_cutoff_idx & r_cutoff_idx
        
        self.BT_mainbranch = mainbranch
        self.BT_sigma = BT_sigma
        self.BT_cut_idx = BT_cut_idx

        if self.display_log == True:
            print("Number of galaxies in velocity and r_max limit : {}".format(r.size))

            print("Data unpacked.")
            print("")

    def find_cluster_center(self, gal_ra, gal_dec, gal_v, cand_mem_idx):

        '''
        Finds the coordinates (RA, Dec, radial vel) of cluster center from the data of candidate member galaxies.
        See Section 4.3 from Serra et al. 2011.

        Parameters
        ----------------------------------
        gal_ra          : numpy ndarray, RA of galaxies (in units of deg)
        gal_dec         : numpy ndarray, Dec of galaxies (in units of deg)
        gal_v           : numpy ndarray, l.o.s. vel of galaxies (in units of km/s)
        cand_mem_idx    : numpy ndarray, values are 1 for candidate members and 0 otherwise

        Returns
        ----------------------------------
        cl_ra   : float, RA of cluster center (in units of deg)
        cl_dec  : float, Dec of cluster center (in units of deg)
        cl_v    : float, radial velocity of cluster center (in units of km/s)
        '''

        # First, consider only the candidate member galaxies.
        cand_gal_ra = gal_ra[cand_mem_idx]                  # RA  of candidate member galaxies
        cand_gal_dec = gal_dec[cand_mem_idx]                # Dec of candidate member galaxies
        cand_gal_v = gal_v[cand_mem_idx]                    # l.o.s. velocity of cnadidate member galaxies

        # Second, calculate the redshift of the cluster center.
        cl_v = np.average(cand_gal_v)                       # radial velocity of the cluster center
        cl_z = cl_v/self.c                                  # redshift of cluster center is calculated as the average of z values of candidate member galaxies

        # Third, calculate the coordinates (RA, Dec) of the cluster center.
        # The coordinates (RA, Dec) of the cluster center are calculated from the peak of number density of candidate member galaxies, using adaptive kernel density estimation
        d_A = astropy.cosmology.LambdaCDM(self.H0, self.Om0, self.Ode0, self.Tcmb0).angular_diameter_distance(cl_z).to(astropy.units.Mpc).value   # angular diamter distance of the cluster center (in units of Mpc); to be used for h_c (smoothing factor)

        #### For adaptive kernel density estimation, we need to calculate various bandwidth factors.
        N = cand_gal_ra.size                                                                                                                # number of candidate members
        h_c = 0.15*d_A/(320 * 100/self.H0)                                                                                                  # smoothing factor for adaptive kernel density estimation; see first paragraph of Section 4.3, Serra et al. 2011
        h_opt = 6.24/(N**(1/6)) * np.sqrt((np.std(cand_gal_ra, ddof=1)**2 + np.std(cand_gal_dec,ddof=1)**2)/2)                              # h_opt  in Eq. 20 of Serra et al. 2011
        gamma = 10**(np.sum(np.log10(self.fq(cand_gal_ra, cand_gal_dec, cand_gal_ra, cand_gal_dec, self.triweight, h_opt)))/N)              # gamma  in the paragraph between Eq. 19 and Eq. 20 of Serra et al. 2011
        lam = (gamma/self.fq(cand_gal_ra, cand_gal_dec, cand_gal_ra, cand_gal_dec, self.triweight, h_opt))**0.5                             # lambda in the paragraph between Eq. 19 and Eq. 20 of Serra et al. 2011
        h = h_c*h_opt*lam                                                                                                                   # local smoothing length
        #### End of bandwidth calculation

        #### Find the peak of the estimated number density on the celestial sphere.
        ra_res = 64                     # resolution    of grid along the RA axis
        ra_min = cand_gal_ra.min()      # minimum value of the grid along the RA axis
        ra_max = cand_gal_ra.max()      # maximum value of the grid along the RA axis

        dec_res = 64                    # resolution    of grid along the Dec axis
        dec_min = cand_gal_dec.min()    # minimum value of the grid along the Dec axis
        dec_max = cand_gal_dec.max()    # maximum value of the grid along the Dec axis

        ra_grid = np.linspace(ra_min, ra_max, ra_res)                                           # grid along the RA  axis
        dec_grid = np.linspace(dec_min, dec_max, dec_res)                                       # grid along the Dec axis

        X, Y = np.meshgrid(ra_grid, dec_grid)                                                   # mesh grid
        den = self.fq(X, Y, cand_gal_ra, cand_gal_dec, self.triweight, h)                       # estimated number density at each point (X, Y)
        max_dec_idx, max_ra_idx = np.unravel_index(np.argmax(den, axis=None), den.shape)        # index of RA grid and Dec grid where the number density is maximum
        cl_ra = ra_grid[max_ra_idx]                                                             # RA  of cluster center
        cl_dec = dec_grid[max_dec_idx]                                                          # Dec of cluster center

        return cl_ra, cl_dec, cl_v                                                              # return value: RA (in units of deg), Dec (in units of deg), and radial velocity (in units of km/s) of the cluster center

    def density_estimation(self, x_data, y_data):

        '''
        Estimates the number density on redshift diagram using adaptive kernel density estimation.

        Parameters
        -----------------------------------
        x_data  : numpy ndarray, 1D array containing rescaled r values of galaxies
        y_data  : numpy ndarray, 1D array containing rescaled v values of galaxies

        Returns
        ----------------------------------
        funtion that returns the estimated number density on redshift diagram at given point (x, y). 
        '''

        # Data points are mirrored to negative r (see the last two sentences from the second paragraph of Section 4.3, Serra et al. 2011).
        x_data_mirrored = np.concatenate((x_data, -x_data))        # mirrored x_data
        y_data_mirrored = np.concatenate((y_data,  y_data))        # mirrored y_data


        # For adaptive kernel density estimation, we need to calculate various bandwidth factors.
        N = x_data.size                                                                                                 # number of data points
        h_opt = 6.24/(N**(1/6)) * np.sqrt((np.std(x_data)**2 + np.std(y_data)**2)/2)                                    # eq. 20 from Serra et al. 2011
        
        gamma = 10**(np.sum(np.log10(self.fq(x_data, y_data, x_data, y_data, self.triweight, h_opt)))/N)                # gamma  defined in Diaferio 1999, between eq. 17 and eq. 18; Here, the term is divided by 2*N because N is the number of original (i.e. un-mirrored) data points
        lam = np.sqrt(gamma/self.fq(x_data, y_data, x_data, y_data, self.triweight, h_opt))                             # lambda defined in Diaferio 1999, between eq. 17 and eq. 18

        fn = lambda h_c: self.M_0(h_c, h_opt, lam, x_data, y_data)                                                      # M_0 function defined in eq. 18 from Diaferio 1999; Here we used the lambda function to fix input arguments other than h_c.
        if self.h_c is None:
            if self.display_log == True:
                print("Calculating h_c.")
            self.h_c = self.find_hc(h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data)                                 # find h_c that minimizes M_0
            if self.display_log == True:
                print("h_c = {:.5e}".format(self.h_c))
        else:
            if self.display_log == True:
                print("User-given h_c = {:.5e}".format(self.h_c))
        self.h_c = self.h_c*self.alpha
        if self.display_log == True:
            print("final value of h_c = {:.5e}".format(self.h_c))

        
        h = self.h_c * h_opt * lam                                                                                      # final h_i (local smoothing length); size of h_i is same as x_data and y_data

        return lambda x, y: self.fq(x, y, x_data, y_data, self.triweight, h)                                            # return value: function that returns the number density on redshift diagram at given point (x, y). 
        
    def find_hc(self, h_opt, lam, x_data, y_data):
        
        '''
        Finds optimal h_cc value for kernel bandwidth used in kernel density estimation.
        Follows the same procedure as CausticApp,
        i.e., starts from hc=0.005 and increments by 0.01 at each step.
        M_0 will decrease at first, and then at some point start to increase.
        hc is incremented until M_0 increases from the previous step.
        The final kappa is determined by locating the minimum of the parabola determined by the 
        last three points.

        Parameters
        -----------------------------
        h_opt       : float,            h_opt calculated as in Eq. 17, Diaferio (1999).
        lam         : float,            lambda calculated as in Diaferio (1999) (below Eq. 17).
        x_data      : numpy ndarray,    1D array containing rescaled r values of galaxies
        y_data      : numpy ndarray,    1D array containing rescaled r values of galaxies

        Returns
        -----------------------------
        Optimal h_c
        '''
        
        init_guess = 0.005
        step_size = 0.01
        max_guess = 155
        
        hc_0 = init_guess
        hc_1 = hc_0 + step_size
        M0_0 = self.M_0(h_c=hc_0, h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data)
        M0_1 = self.M_0(h_c=hc_1, h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data)
        if(self.display_log):
            print("Iteration   1, hc = {:.7f}: M_0 = {:.7e}".format(hc_0, M0_0))
            print("Iteration   2, hc = {:.7f}: M_0 = {:.7e}".format(hc_1, M0_1))
        
        i = 2
        while(i < max_guess):
            hc_2 = hc_1 + step_size
            M0_2 = self.M_0(h_c=hc_2, h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data)
            
            if(self.display_log):
                print("Iteration {:3}, hc = {:.7f}: M_0 = {:.7e}".format(i+1, hc_2, M0_2))
            
            if (M0_2 > M0_1):
                # Find minimum by parabolic 
                h_c = self.parabola_min(hc_0, hc_1, hc_2, M0_0, M0_1, M0_2)
                
                if(self.display_log):
                    print("Optimal h_c: {}".format(h_c))
                
                return h_c
                
            else:
                i += 1
                hc_0 = hc_1
                hc_1 = hc_2
                M0_0 = M0_1
                M0_1 = M0_2
        
        raise Exception("Failed to find optimal kernel size hc.")

    def parabola_min(self, x0, x1, x2, y0, y1, y2):
        
        '''
        Finds the minimum point of the parabola that passes three points (x0, y0), (x1, y1), and (x2, y2)
        
        Parameters
        -----------------------------------
        x0  : float,    x coordinate of first  point
        x1  : float,    x coordinate of second point
        x2  : float,    x coordiante of third  point
        y0  : float,    y coordinate of first  point
        y1  : float,    y coordinate of second point
        y2  : float,    y coordiante of third  point

        Returns
        ----------------------------------
        x coordinate of the minimum point of the parabola

        '''
        
        k0 = y0 / ((x0 - x1)*(x0 - x2))
        k1 = y1 / ((x1 - x0)*(x1 - x2))
        k2 = y2 / ((x2 - x0)*(x2 - x1))
        
        xmin = (k0*(x1 + x2) + k1*(x0 + x2) + k2*(x0 + x1)) / (2*(k0+k1+k2))
        return xmin

    def M_0(self, h_c, h_opt, lam, x_data, y_data):

        '''
        Cost function which h_c should minimize (eq. 18 from Diaferio 1999).

        Parameters
        -----------------------------------
        h_c     : float,            smoothing factor h_c
        h_opt   : float,            optimal smoothing length (eq. 20 from Serra et al. 2011)
        lam     : numpy ndarray,    local smoothing length; same size as x_data and y_data
        x_data  : numpy ndarray,    1D array containing rescaled r values of galaxies
        y_data  : numpy ndarray,    1D array containing rescaled v values of galaxies

        Returns
        ----------------------------------
        Value of cost function evaluated for given h_c     
        '''

        h = h_c * h_opt * lam                               # local smoothing length
        N = x_data.size                                     # number of data points

        # calculating the first term
        #### set up grids for numerical integration; x_grid and y_grid here is different from those used in the main function Caustics().
        #### Note that the value of triweight funtion is 0 outside (x/h)**1 + (y/h)**1 = 1
        x_min = 0                                           # minimum value in the x_grid
        x_max = np.max(x_data + h)                     # maximum value in the x_grid
        y_min = np.min(y_data - h)                     # minimum value in the y_grid
        y_max = np.max(y_data + h)                     # maximum value in the y_grid
        
        x_res = self.r_res                             # resolution of x_grid
        y_res = self.v_res                             # resolution of y_grid

        x_grid = np.linspace(x_min, x_max, x_res)      # grid along rescaled r-axis (x-axis) used for numerical integration
        y_grid = np.linspace(y_min, y_max, y_res)      # grid along rescaled v-axis (y-axis) used for numerical integration

        X, Y = np.meshgrid(x_grid, y_grid)             # mesh grid
        
        f_squared = self.fq(X, Y, x_data, y_data, self.triweight, h)**2                         # squared value of fq calcuated at each point (X, Y)

        term_1 = np.trapz(np.trapz(f_squared, x=y_grid, axis=0), x=x_grid)                      # first term of M_0 is the integration of fq squared
        
        # calculating the second term (refer to eq. 3.37 and Section 5.3.4 of Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London)
        # x_pairs = np.subtract.outer(x_data, x_data)                    # 2D array of size (N, N); element (i, j) is x_data[i]-x_data[j] i.e. pair-wise subtraction of x_data and x_data
        # y_pairs = np.subtract.outer(y_data, y_data)                    # 2D array of size (N, N); element (i, j) is y_data[i]-y_data[j] i.e. pair-wise subtraction of y_data and y_data
        # a = np.sum(self.triweight(x_pairs/h, y_pairs/h) / (h**2))      # sum of fq evaluated at all data points (x, y)
        # b = self.triweight(0, 0) * np.sum(1/(h**2))
        # term_2 = 2/(N*(N - 1)) * (a - b)

        term_2 = 0
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            term_2 += self.fq(x=x_data[i], y=y_data[i], x_data=x_data[mask], y_data=y_data[mask], K=self.triweight, h=h[mask])
        term_2 *= 2/N
        
        return term_1 - term_2                                              # return value: M_0 function evaluated for given h_c

    def triweight(self, x, y):

        '''
        2D triwegith kernel

        Parameters
        -----------------------------------
        x   : float, x coordinate 
        y   : float, y coordinate

        Returns
        ----------------------------------
        Value of triweight kernel evaulated at (x, y).     
        '''

        t = np.sqrt(x**2 + y**2)
        return 4/np.pi * ((1-t**2)**3) * (t < 1)        # return 4/pi * (1-t**2)**3 if t<1; otherwise 0        

    def fq(self, x, y, x_data, y_data, K, h):

        '''
        Estimated number density function, using kernel density estimation.
        
        Parameters
        ---------------------------------------
        x       : float or numpy ndarray,   x-axis value(s) of the point(s) where the density is to be calculated
        y       : float or numpy ndarray,   y-axis value(s) of the point(s) where the density is to be calculated
        x_data  : numpy ndarray,            x-axis value of the observed data points
        y_data  : numpy ndarray,            y-axis value of the observed data points
        K       : function,                 kernel function
        h       : float or numpy ndarray,   bandwidth of the kernel
        
        Returns
        ---------------------------------------
        Estimated number density value at the given point (x, y).
        '''

        N = x_data.size                                                     # number of data points
        
        x_pairs = np.subtract.outer(x, x_data)                              # (M+N)-D array, where M is the dimension of x and N is the dimension of x_data; pair-wise subtraction of x and x_data
        y_pairs = np.subtract.outer(y, y_data)                              # (M+N)-D array, where M is the dimension of y and N is the dimension of y_data; pair-wise subtraction of y and y_data
        
        x_data_mirrored = np.concatenate([x_data, -x_data])
        y_data_mirrored = np.concatenate([y_data,  y_data])
        if np.size(h) != 1:
            h = np.concatenate([h, h])
        
        x_pairs = np.subtract.outer(x, x_data_mirrored)                              # (M+N)-D array, where M is the dimension of x and N is the dimension of x_data; pair-wise subtraction of x and x_data
        y_pairs = np.subtract.outer(y, y_data_mirrored)                              # (M+N)-D array, where M is the dimension of y and N is the dimension of y_data; pair-wise subtraction of y and y_data
        
        return np.sum(K(x_pairs/h, y_pairs/h)/(h**2), axis = -1) / (2*N)        # return value: estimated number density value at given point (x, y); Summation must be done for x_data and y_data. Because the the dimension of x and y may vary, axis=-1 is used.

    def S(self, kappa, r_grid, v_grid, r_res, den, R, vvar):

        '''
        Cost function which kappa should minimize; defined in eq. 22 of Serra et al. 2011

        Parameters
        ---------------------------------------
        kappa   :   float,  density level where the contour is to be drawn 
        r_grid  :   numpy ndarray,  grid along the r-axis
        v_grid  :   numpy ndarray,  grid along the v-axis
        r_res   :   int,            resolution of teh r_axis
        den     :   numpy ndarray,  2D array of number density calculated for r_grid and v_grid
        R       :   float,          mean projected distance to the center from each candidate member galaxy (in units of Mpc)
        vvar    :   float,          variance of relative l.o.s. velocity of candidate member galaxies (in units of (km/s)^2)

        Returns
        ---------------------------------------
        Value of the cost function.
        '''
        A = self.calculate_A(kappa, den, r_grid, v_grid)                        # amplitude of the caustic lines for given kappa
        
        # phi is calculated by integrating fq within the caustic lines.
        phi = np.empty(r_res)                                                   # initialize phi as an empty array with the same size as the r_grid
        for i in range(phi.size):                                               # for each grid points in r_grid, calculate phi
            phi[i] = np.trapz(den[(v_grid < A[i]) & (v_grid > -A[i]), i], x = v_grid[(v_grid < A[i]) & (v_grid > -A[i])])          # 1D integration of of number density fixed at r_grid[i], and integration range is only within the caustic lines

        if np.trapz(phi[r_grid < R], x = r_grid[r_grid < R]) == 0:              # If the integration of phi along r_grid is 0, this causes divided by zero. Thus, we just return infinity to avoid warning/error.
            return np.inf

        v_esc_mean_squared = np.trapz((A[r_grid < R]**2) * phi[r_grid < R], x = r_grid[r_grid < R]) / np.trapz(phi[r_grid < R], x = r_grid[r_grid < R])       # mean squared escape velocity
        
        v_mean_squared = vvar                                                   # mean squared velocity (calculated in advance from candidate member galxies)
        print("kappa = {:.6e}: v^2 = {:.6e}, v_esc^2 = {:.6e}".format(kappa, np.sqrt(v_mean_squared), np.sqrt(v_esc_mean_squared)))
        return abs(v_esc_mean_squared - 4*v_mean_squared)                       # return value: absolute value of the difference of mean sqaured escape velocity and mean squared velocity

    def find_kappa(self, r_grid, v_grid, den):
        
        '''
        Finds optimal kappa value for threshold by minimizing S(kappa).
        Follows the same procedure as CausticApp,
        i.e., starts from kappa=0 and increments kappa linearly.
        S(kappa) will decrease at first, and then at some point start to increase.
        kappa is incremented until S(kappa) increases from the previous step.
        The final kappa is determined by locating the minimum of the parabola determined by the 
        last three points.

        Parameters
        -----------------------------
        r_grid      : numpy ndarray, grid along r-axis (1D array)
        v_grid      : numpy ndarray, grid along v-axis (1D array)
        den         : numpy ndarray, number density of galaxies in the redshift space (2D array)

        Returns
        -----------------------------
        Optimal kappa
        '''
        
        fn = lambda kappa: self.S(kappa, r_grid, v_grid, self.r_res, den, self.R, self.vvar)
        k_min = 0
        k_max = den.max()
        nstep = 51
        step_size = (k_max - k_min) / nstep
        
        k0 = k_min
        k1 = k0 + step_size
        S0 = fn(k0)
        S1 = fn(k1)
        
        i = 2
        while(i < nstep):
            k2 = k1 + step_size
            S2 = fn(k2)
            
            if S2 > S1:
                k = self.parabola_min(k0, k1, k2, S0, S1, S2)
                return k
            
            else:
                k0 = k1
                k1 = k2
                S0 = S1
                S1 = S2
                i += 1

    def calculate_A(self, kappa, den, r_grid, v_grid):

        '''
        Calculates the caustic amplitude for given kappa.

        Parameters
        --------------------------------------
        kappa   : float,            level at which the contour is drawn
        den     : numpy ndarray,    2D array of estimated number density calculated for r_grid and v_grid
        r_grid  : numpy ndarray,    grid along the r-axis
        v_grid  : numpy ndarray,    grid along the v-axis

        Returns
        --------------------------------------
        caustic amplitude at each point in r_grid
        '''
        contours = skimage.measure.find_contours(den, kappa)                    # contour lines of number density for given level kappa

        v_step = v_grid[1] - v_grid[0]                                          # step size of v_grid

        A = np.full(r_grid.size, max(abs(self.v_max), abs(self.v_min)))         # amplitude of the caustic lines determined at each grid value of r_grid; initialize with the maximum absolute value of the v_grid
        zero_idx = None                                                         # index along the r_grid from which A should be 0; calculated by finding when the contour line crosses v = 0
        for contour in contours:                                                # contours is a list of contour, as there may be multiple contour lines for a single level
            r_cont = contour[:,1]                                               # r_grid indices of the contour line
            v_cont = contour[:,0]                                               # v_grid indices of the contour line

            if not (r_cont[0] == r_cont[-1] and v_cont[0] == v_cont[-1]):       # If the contour line does NOT loop back to itself,
                check_sign = True                                               # then we need to check when the sign of v_grid changes along the contour (i.e. when the contour line crosses v = 0).
            else:                                                               # If the contour line loops back to itself,
                check_sign = False                                              # we do not need to check when the sign of v_grid changes along the contour (i.e. when the contour line crosses v = 0).

            if v_cont[0]*v_step + v_grid.min() < 0:                             # The contour found by skimage.measure goes CCW; i.e., for non-looping, main contours, it generally will start from r=0, v<0 and end at r=0, v>0                             
                negative = True                                                 # If so, mark it as negative = True, to show that the starting point is at v<0.
            else:                                                               # If not,
                negative = False                                                # mark it as negative = False.
            
            int_idx = (r_cont == r_cont.astype(int))                            # The indices given by skimage.measure includes 'float' indices, which are interpolated values. We only need interger indices along the r_grid.
            r_cont_grid = r_cont[int_idx].astype(int)                           # indices of the contour line on r_grid, using only integer indices along the r_grid
            v_cont_grid = v_cont[int_idx]                                       # indices of the contour line on v_grid, using only integer indices along the r_grid (v_grid indices need not be integers)
            for r_idx, v_idx in zip(r_cont_grid, v_cont_grid):                  # for coordinate indices (r_idx, v_idx) on the contour line
                v = v_idx*v_step + v_grid.min()                                 # v value on the v_grid corresponding to v_idx
                if check_sign and negative and (v >= 0):                        # 1) If we need to check the sign change of v (i.e., the contour is non-looping), 2) if the contour started from negative v, and 3) if the current v value is positive, then we have found where the contour line crosses v = 0. 
                    zero_idx = r_idx+1                                          # In this case, all values of A should be 0 from r_idx + 1.
                    negative = False                                            # And as the v value has became positive, set negative to False.
                A[r_idx] = min(A[r_idx], abs(v))                                # Caustic line amplitude is the minimum value of abs(v) for given r.

        if zero_idx != None:                                                    # zero_idx is initialized as None before the loop. Thus, if zero_idx is NOT None, then there is a point where the contour line crossed v = 0.
            A[zero_idx:] = 0                                                    # A(r) is set to 0 for r >= r_grid[zero_idx].

        A = self.grad_restrict(A, r_grid, grad_limit=self.grad_limit)           # restrict gradient of A
        
        return A                                                                # return value: amplitude of caustic line determined at grid points on r_grid

    def grad_restrict(self, A, r, grad_limit, new_grad = 0.25):

        '''
        Restricts the gradient of A.
        See Section 4.3 from Serra et al. 2011.

        Parameters
        -----------------------------
        A           : numpy ndarray,    amplitude of caustic line 
        r           : numpy ndarray,    grid values along the r axis
        grad_limit  : float,            upper limit of gradient
        new_grad    : float,            new gradient which the replaced amplitude should satisfy

        Returns
        -----------------------------
        amplitude of caustic line with gradient restriction applied
        '''

        for i in range(A.size-1):                               # iterate except for the last element of A
            if (A[i] <= 0) or (A[i+1] <= 0) or (r[i] == 0):     # If any of A[i], A[i+1], or r[i] is 0, then it will cause a warning when given to log.
                continue
            
            else:
                log_grad = (np.log(A[i+1])-np.log(A[i])) / (np.log(r[i+1]) - np.log(r[i]))          # gradient of the current A
                if log_grad > grad_limit:
                    A[i+1] = np.exp( np.log(A[i]) + new_grad*(np.log(r[i+1]) - np.log(r[i])) )      # new value of A[i+1]
                
        return A

    def membership(self, r, v, r_grid, A):

        '''
        Determines which galaxies are inside the caustic lines

        Parameters
        -----------------------------
        r       :   numpy ndarray, projected distance from the cluster center to each galaxy (in units of Mpc)
        v       :   numpy ndarray, relative l.o.s. velocity of each galaxy (in units of km/s)
        r_grid  :   numpy ndarray, grid along the r-axis
        A       :   numpy ndarray, amplitude of caustic line

        Returns
        -----------------------------
        bool array, 1 for member galaxies and 0 for interlopers
        '''

        A_at_r = np.interp(r, r_grid, A)
        return abs(v) < A_at_r
    
    def minimize_fn(self, fn, a, b, positive = False, it = 0):

        '''
        Minimizes given function fn using golden section search.

        Parameters
        -----------------------------
        fn          : function, function to minimize
        a           : float,    lower bound of input to fn
        b           : float,    upper bound of input to fn
        positive    : bool,     True if search range should be constrained to positive values; default value False
        it          : int,      number of iterations; used only for recursion of this function

        Returns
        -----------------------------
        Value that minimizes fn.
        '''
        
        if self.display_log == True:
            print("search range: {} ~ {}".format(a, b))
        
        it += 1                                                                 # number of iterations
        
        if positive:                                                            # If the search range should be restricted to positive values,
            a = max(a, 0)                                                       # a is set to 0 if a < 0.
            b = max(b, 0)                                                       # b is set to 0 if b < 0.
            if (a == 0) and (b == 0):                                           # If a == 0 and b == 0, it means both a and b given are non-positive values.
                raise Exception("Cannot optimize the function. Either reset the search range, or check if the function is optimizable.")
        
        a_init = a                                                              # initial value of a
        b_init = b                                                              # initial value of b
        
        R = (np.sqrt(5)-1)*0.5                                                  # golden ratio
        
        TOL = 1e-6 * abs(b - a)                                                 # TOLerance
        if TOL == 0:
            TOL = 1e-5
        
        error = TOL + 1                                                         # initial error set to an arbitrarily larger value than TOL
        
        x1 = b - R*(b - a)                                                      # initial value to guess
        x2 = a + R*(b - a)                                                      # initial value to guess
        
        f1 = fn(x1)                                                             # initial function value at x1
        f2 = fn(x2)                                                             # initial function value at
        
        while error > TOL:                                                      # iterate until the error ( = abs(b-a)) becomes smaller than tolerance
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + R*(b - a)
                f2 = fn(x2)
                
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - R*(b - a)
                f1 = fn(x1)

            error = abs(b - a)
            
            if self.display_log == True:
                print("iteration : {}".format(it), end = '\r')
            
            it += 1
            if it > 100:
                raise Exception("Cannot optimize the function. Either reset the search range, or check if the function is optimizable.")

        if self.display_log == True:
            print("iteration : {}".format(it))
        if abs(a_init - a) < TOL:                                           # If a is very close to the inital value a_init, then this means that minimum value is on the left side of the search range.
            a = a_init - (b_init - a_init)
            b = a_init
            
            if self.display_log == True:
                print("Shifting search range to ({}, {})".format(a, b))
            return self.minimize_fn(fn, a, b, positive, it)
            
        elif abs(b_init - b) < TOL:                                         # If b is very close to the inital value b_init, then this means that minimum value is on the right side of the search range.
            a = b_init
            b = b_init + (b_init - a_init)
            if self.display_log == True:
                print("Shifting search range to ({}, {})".format(a, b))
            return self.minimize_fn(fn, a, b, positive, it)
    
        return a