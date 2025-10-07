# This class is for applying caustic method as presented by Diaferio 1999 and Serra et al. 2011.
# When a list of galaxy data containing RA, Dec, and l.o.s. velocity is given, this class calculates the caustic line and determines the membership of each galaxy.
# The code also calculates the mass profile of the cluster.

# Code written by Wooseok Kang, Astronomy Program, Dept. of Physics and Astronomy, Seoul National University (SNU)
# with help from members of Exgalcos team, SNU.
# If you have any questions or comments, contact woodykang@snu.ac.kr

# References:
# 1. Diaferio, A. 1999, MNRAS, 309, 610
# 2. Gifford, D., Miller, C., & Kern, N. 2013, ApJ, 773, 116
# 3. Serra, A. L., Diaferio, A., Murante, G., & Borgani, S. 2011, MNRAS, 412, 800
# 4. Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London

import numpy as np
from astropy.coordinates import angular_separation, SkyCoord
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import LambdaCDM
import skimage.measure
from .hierarchical_clustering import hier_clustering
from .kde import density_estimation, triweight, fq, parabola_min

def run_from_file(fpath,
                  v_lower,
                  v_upper,
                  v_max,
                  r_max,
                  center_given=False,
                  H0=100,
                  Om0=0.3,
                  Ode0=0.7,
                  Tcmb0=2.7,
                  q=25,
                  r_res=100,
                  v_res=100,
                  BT_thr="ALS",
                  gal_m=1e12,
                  h_c = None,
                  kappa=None,
                  alpha=1,
                  grad_limit=2,
                  F_b=0.7,
                  sig_pl=None,
                  display_log=True):
    
    '''
    Calculates the caustics by reading the input file.

    Parameters
    ----------------------------------
    fpath           : path to the input file
    v_lower         : float, lower line-of-sight velocity limit of galaxies to use in the caustics (in units of km/s); velocity relative to the observer (cz)
    v_upper         : float, upper line-of-sight velocity limit of galaxies to use in the caustics (in units of km/s); velocity relative to the observer (cz)
    v_max           : float, maximum line-of-sight velocity relative to the cluster center when drawing the caustics diagram (in units of km/s); the minimum value is automatically set as -v_max
    r_max           : float, maximum projected distance from the cluster center when drawing the caustics diagram (in units of Mpc); the minimum projected distance is trivially zero
    center_given    : bool, optional, whether to use the cluster center (RA, decl, v); default value False
    H0              : float, optional, Hubble parameter at the current Universe (in units of km/s/Mpc); default value 100
    Om0             : float, optional, matter density parameter at the current Universe; default value 0.3
    Ode0            : float, optional, dark energy density parameter at the current Universe; default value 0.7
    Tcmb0           : float, optional, temperature of the cosmic microwave background at the current Universe (in units of K); default value 2.7
    q               : float, optional, rescaling factor defined in Diaferio 1999; default value 25
    r_res           : float, optional, number of grid points along the projected distance in the caustics diagram; default value 100
    v_res           : float, optional, number of grid points along the line-of-sight velocity in the caustics diagram; default value 100
    BT_thr          : str, optional, binary tree threshold method; only supports "ALS" at this moment; default value "ALS"
    gal_m           : float, optional, mass of galaxies to be used in the binary tree construction (in units of solar mass); default value 1e12
    h_c             : float or None, optional, h_c value for kernel bandwidth used in kernel density estimation; if given, uses the given h_c instead of calculating; default value None
    kappa           : float or None, optional, number density threshold for determining the caustics amplitude; if given, uses the value instead of calculating; default value None
    alpha           : float, optional, smoothing parameter for kernel density estimation of the redshift space, multiplied to the kernel bandwidth; default value 1
    grad_limit      : float, optional, limit of the log-gradient of the caustic amplitude, default value 2
    F_b             : float, optional, filling factor used when calculating mass profile; default value 0.7
    sig_pl          : float or None, optional, sigma plateau (in units of km/s); if given, uses the value instead of calculating the sigma plateau; default value None,
    display_log     : bool, optional, prints log if True; default value True

    Returns
    ----------------------------------
    results         : CausticResults, contains information of the caustics results
    '''
    
    ra_gal, dec_gal, v_gal = np.loadtxt(fpath, skiprows=1, unpack=True)   # RA (deg), Dec (deg), l.o.s velocity (km/s)
    cluster_data = np.loadtxt(fpath, max_rows=1)                          # In the first row, the format is N, cl_ra, cl_dec, cl_v
                                                                                # where N is number of galaxies observed, and
                                                                                # cl_ra, cl_dec, cl_z are RA (deg), DEC (deg), los velocity (km/s) of the cluster (if specified)
    ra_cl = None
    dec_cl = None
    v_cl = None
    if center_given == True:
        if np.size(cluster_data) == 1:
            raise Exception("First row of input file has only one number, while the center coordianates should be given.")
        N = int(cluster_data[0])                                                # number of galaxies given as input
        ra_cl = cluster_data[1]
        dec_cl = cluster_data[2]
        v_cl = cluster_data[3]
        
    else:
        if np.size(cluster_data) == 1:
            N = int(cluster_data)
        else:
            N = int(cluster_data[0])
    
    if N != ra_gal.size:                                                        # emit error if N does not match the actual number of galaxies listed in the file
        raise Exception("Number of galaxies stated in the first line of file does not match the number of galaxies listed.")

    result = run_from_array(ra_gal, dec_gal, v_gal, v_lower, v_upper, r_max, v_max, ra_cl, dec_cl, v_cl, center_given, H0, Om0, Ode0, Tcmb0, q, r_res, v_res, BT_thr, gal_m, h_c, kappa, alpha, grad_limit, F_b, sig_pl, display_log)
    return result

def run_from_array(ra_gal,
                   dec_gal,
                   v_gal,
                   v_lower,
                   v_upper,
                   r_max,
                   v_max,
                   ra_cl=None,
                   dec_cl=None,
                   v_cl=None,
                   center_given=False,
                   H0=100,
                   Om0=0.3,
                   Ode0=0.7,
                   Tcmb0=2.7,
                   q=25,
                   r_res=100,
                   v_res=100,
                   BT_thr="ALS",
                   gal_m=1e12,
                   h_c = None,
                   kappa=None,
                   alpha=1,
                   grad_limit=2,
                   F_b=0.7,
                   sig_pl=None,
                   display_log=True):
    
    '''
    Calculates the caustics from input numpy arrays.

    Parameters
    ----------------------------------
    ra_gal          : numpy ndarray, array of right ascensions of input galaxies (in units of degree)
    dec_gal         : numpy ndarray, array of declinations of input galaxies (in units of degree); must be the same size as ra_gal
    v_gal           : numpy ndarray, array of line-of-sight velocities of input galaxies relative to the observer (cz in units of km/s); must be the same size as ra_gal
    v_lower         : float, lower line-of-sight velocity limit of galaxies to use in the caustics (in units of km/s); velocity relative to the observer (cz)
    v_upper         : float, upper line-of-sight velocity limit of galaxies to use in the caustics (in units of km/s); velocity relative to the observer (cz)
    v_max           : float, maximum line-of-sight velocity relative to the cluster center when drawing the caustics diagram (in units of km/s); the minimum value is automatically set as -v_max
    r_max           : float, maximum projected distance from the cluster center when drawing the caustics diagram (in units of Mpc); the minimum projected distance is trivially zero
    center_given    : bool, optional, whether to use the cluster center (RA, decl, v); default value False
    H0              : float, optional, Hubble parameter at the current Universe (in units of km/s/Mpc); default value 100
    Om0             : float, optional, matter density parameter at the current Universe; default value 0.3
    Ode0            : float, optional, dark energy density parameter at the current Universe; default value 0.7
    Tcmb0           : float, optional, temperature of the cosmic microwave background at the current Universe (in units of K); default value 2.7
    q               : float, optional, rescaling factor defined in Diaferio 1999; default value 25
    r_res           : float, optional, number of grid points along the projected distance in the caustics diagram; default value 100
    v_res           : float, optional, number of grid points along the line-of-sight velocity in the caustics diagram; default value 100
    BT_thr          : str, optional, binary tree threshold method; only supports "ALS" at this moment; default value "ALS"
    gal_m           : float, optional, mass of galaxies to be used in the binary tree construction (in units of solar mass); default value 1e12
    h_c             : float or None, optional, h_c value for kernel bandwidth used in kernel density estimation; if given, uses the given h_c instead of calculating; default value None
    kappa           : float or None, optional, number density threshold for determining the caustics amplitude; if given, uses the value instead of calculating; default value None
    alpha           : float, optional, smoothing parameter for kernel density estimation of the redshift space, multiplied to the kernel bandwidth; default value 1
    grad_limit      : float, optional, limit of the log-gradient of the caustic amplitude, default value 2
    F_b             : float, optional, filling factor used when calculating mass profile; default value 0.7
    sig_pl          : float or None, optional, sigma plateau (in units of km/s); if given, uses the value instead of calculating the sigma plateau; default value None,
    display_log     : bool, optional, prints log if True; default value True

    Returns
    ----------------------------------
    results         : CausticResults, contains information of the caustics results
    '''
    
    c = 299792.458  # speed of light in [km/s]

# safety check
    if (ra_gal.size != dec_gal.size) | (ra_gal.size != v_gal.size):
        raise Exception("Input data must be of same length.")
    
    if ((ra_cl is None) or (dec_cl is None) or (v_cl is None)) and center_given:
        raise Exception("Cluster center must be given when center_given=True.")
    

# find candidate members from hierarchical clustering
    within_vrange = (v_lower < v_gal) & (v_gal < v_upper)
    # BT_sigma value should be divided by (1+z_cl) in the later part of the code. For now, we don't know the cluster redshift.
    cand_mem_idx, mainbranch, BT_sigma, BT_cut_idx = hier_clustering(ra_gal, dec_gal, v_gal, mask=within_vrange, gal_m=gal_m, sig_pl=sig_pl)
    if display_log == True:
        print("Hierarchical clustering done.")
        print("Number of candidate members : {}".format(sum(cand_mem_idx)))
        print("")


# find cluster center from candidate members
    ra_cl_cand, dec_cl_cand, v_cl_cand = find_cluster_center(ra_gal, dec_gal, v_gal, cand_mem_idx, H0, Om0, Ode0, Tcmb0)
    z_cl_cand = v_cl_cand/c
    
    if not center_given:            # set cluster center as that found from candidate members
        ra_cl = ra_cl_cand
        dec_cl = dec_cl_cand
        v_cl = v_cl_cand
    z_cl = v_cl/c

    if display_log == True:
        print("Cluster center found in this code: RA = {:4.5f} deg, Dec = {:4.5f} deg, v = {:.0f} km/s, z = {:6f}".format(ra_cl_cand, dec_cl_cand, v_cl_cand, z_cl_cand))
        if center_given:
            print("Using cluster center given by user.")
            print("Cluster center: RA = {:4.5f} deg, Dec = {:4.5f} deg, v = {:.0f} km/s, z = {:6f}".format(ra_cl, dec_cl, v_cl, z_cl))
            print("")
        else:
            print("Using cluster center calculated here.")

    BT_sigma = BT_sigma/(1+z_cl)        # correct for cosmological redshift


# calculate projected clustercentric distance and l.o.s. velocity
    LCDM = LambdaCDM(H0, Om0, Ode0, Tcmb0)                          # assumed cosmology
    d_A = LCDM.angular_diameter_distance(z_cl).to_value(u.Mpc)      # angular diameter distance at the cluster redshift
    co_gal = SkyCoord(ra=ra_gal, dec=dec_gal, unit=u.deg)           # coordinates of galaxies
    co_cl = SkyCoord(ra=ra_cl, dec=dec_cl, unit=u.deg)              # coordiante of cluster center
    angle_sep = co_cl.separation(co_gal).to_value(u.rad)            # angular separation of galaxies from the cluster center [rad]
    
    r = np.sin(angle_sep)*d_A                                       # projected clustercentric distance of galaxies
    v = (v_gal - v_cl)/(1+z_cl)                                     # line-of-sight velocity of galaxies

    R_avg = np.average(r[cand_mem_idx])                             # average projected distance from the center of the cluster to candidate member galaxies (in units of Mpc); later to be used for function S(k)
    v_var = np.average(v[cand_mem_idx]**2)                          # variance of v calculated from candidate members (in units of (km/s)**2); later to be used for function S(k)
    if display_log == True:
        print("Mean clustercentric distance of candidate members:   {:4.5f} Mpc".format(R_avg))
        print("Velocity Dispersion of candidate members:            {:4.5f} km/s".format(np.sqrt(v_var)))
        print("")

    within_rrange = (r < r_max)

    if display_log == True:
        print("Number of galaxies in velocity and r_max limit : {}".format(np.sum(within_rrange & within_vrange)))

    
# estimate number density in redshift space
    r_min = 0                                                   # minimum r is, obviously, 0
    r_grid = np.linspace( r_min, r_max, r_res)                  # grid of r-axis on the redshift diagram
    v_grid = np.linspace(-v_max, v_max, v_res)                  # grid of v-axis on the redshift diagram
    
    # rescaled values
    x_data = r[within_rrange & within_vrange]*H0                # rescaled data points in r-axis (in units of km/s)
    y_data = v[within_rrange & within_vrange]/q                 # rescaled data points in v-axis (in units of km/s)
    x_grid = r_grid*H0                                          # grid of rescaled r-axis (in units of km/s)
    y_grid = v_grid/q                                           # grid of rescaled v-axis (in units of km/s)

    # adaptive kernel density estimation
    if display_log == True:
        print("Estimating number density.")
    
    f_den = density_estimation(x_data, y_data, r_res, v_res, alpha, h_c, display_log)      # function that returns the number density on the redshift diagram when coordianates (x, y) are given

    if display_log == True:
        print("Number density estimation done.\n")

    X, Y = np.meshgrid(x_grid, y_grid)
    den = f_den(X, Y)*H0/q*2                                    # number density estimated at each point X, Y, normalized to be 1 when integrated along r, v axis
    
# find optimal kappa
    if kappa is None:
        if display_log == True:
            print("Minimizing S(kappa) to find kappa.")
        kappa = find_kappa(r_grid, v_grid, r_res, den, R_avg, v_var, grad_limit)
        if display_log == True:
            print("kappa found. kappa={:.5e}\n".format(kappa))
    else:
        if display_log == True:
            print("Using user input for kappa={:.5e}, S(kappa)={:.5e}\n".format(kappa, S(kappa, r_grid, v_grid, r_res, den, R_avg, v_var, grad_limit)))
    
# calculate A(r) with the minimized kappa
    A, dA = calculate_A(kappa, den, r_grid, v_grid, grad_limit)
    if display_log == True:
        print("Caustic amplitude calcualted.")

# calculate enclosed mass profile with A
    M, dM = calculate_M(r_grid, A, dA, F_b)
    if display_log == True:
        print("Mass profile calculation done.\n")

# determine membership
    member = membership(r, v, r_grid, A)                        # numpy array where the values are 1 for members and 0 for interlopers
    if display_log == True:
        print("Membership determination done.\n")

# calculate velocity dispersion of member galaxies
    vdisp, vdisp_err = velocity_dispersion(v[member], give_error=True)
    if display_log == True:
        print("")
        print("")
        print("============== Summary ==============")
        print("Cluster center found with candidate members: RA={:.3f} deg, Dec={:.3f} deg, v={:.0f} km/s".format(ra_cl_cand, dec_cl_cand, v_cl_cand))
        print("Number of members: {} / {}".format(np.sum(member), len(member)))
        print("Velocity dispersion of member galaxies: {:.0f} +- {:.0f} km/s".format(vdisp, vdisp_err))
        print("Mass of the cluster: {:.2e} M_sun".format(np.max(M)))


    result = CausticResults(ra_gal=ra_gal, dec_gal=dec_gal, v_gal=v_gal, r=r, v=v, member=member, ra_cl=ra_cl, dec_cl=dec_cl, v_cl=v_cl, ra_cl_cand=ra_cl_cand, dec_cl_cand=dec_cl_cand, v_cl_cand=v_cl_cand, vdisp=vdisp, vdisp_err=vdisp_err, r_grid=r_grid, v_grid=v_grid, A=A, dA=dA, M=M, dM=dM, BT_mainbranch=mainbranch, BT_sigma=BT_sigma, BT_cut_idx=BT_cut_idx)
    return result
    

def find_cluster_center(ra_gal, dec_gal, v_gal, cand_mem_idx, H0, Om0, Ode0, Tcmb0):
        c = 299792.458
        '''
        Finds the coordinates (RA, Dec, radial vel) of cluster center from the data of candidate member galaxies.
        See Section 4.3 from Serra et al. 2011.

        Parameters
        ----------------------------------
        ra_gal          : numpy ndarray, RA of galaxies (in units of deg)
        dec_gal         : numpy ndarray, Dec of galaxies (in units of deg)
        v_gal           : numpy ndarray, l.o.s. vel of galaxies (in units of km/s)
        cand_mem_idx    : numpy ndarray, values are 1 for candidate members and 0 otherwise

        Returns
        ----------------------------------
        ra_cl   : float, RA of cluster center (in units of deg)
        dec_cl  : float, Dec of cluster center (in units of deg)
        v_cl    : float, radial velocity of cluster center (in units of km/s)
        '''

        # First, consider only the candidate member galaxies.
        cand_gal_ra = ra_gal[cand_mem_idx]                  # RA  of candidate member galaxies
        cand_gal_dec = dec_gal[cand_mem_idx]                # Dec of candidate member galaxies
        cand_gal_v = v_gal[cand_mem_idx]                    # l.o.s. velocity of cnadidate member galaxies

        # Second, calculate the redshift of the cluster center.
        v_cl = np.average(cand_gal_v)                       # radial velocity of the cluster center
        z_cl = v_cl/c                                  # redshift of cluster center is calculated as the average of z values of candidate member galaxies

        # Third, calculate the coordinates (RA, Dec) of the cluster center.
        # The coordinates (RA, Dec) of the cluster center are calculated from the peak of number density of candidate member galaxies, using adaptive kernel density estimation
        LCDM = LambdaCDM(H0, Om0, Ode0, Tcmb0)
        d_A = LCDM.angular_diameter_distance(z_cl).to_value(u.Mpc)   # angular diamter distance of the cluster center (in units of Mpc); to be used for h_c (smoothing factor)

        #### For adaptive kernel density estimation, we need to calculate various bandwidth factors.
        N = cand_gal_ra.size                                                                                                      # number of candidate members
        h_c = 0.15*d_A/(320 * 100/H0)                                                                                             # smoothing factor for adaptive kernel density estimation; see first paragraph of Section 4.3, Serra et al. 2011
        h_opt = 6.24/(N**(1/6)) * np.sqrt((np.std(cand_gal_ra, ddof=1)**2 + np.std(cand_gal_dec,ddof=1)**2)/2)                    # h_opt  in Eq. 20 of Serra et al. 2011
        gamma = 10**(np.sum(np.log10(fq(cand_gal_ra, cand_gal_dec, cand_gal_ra, cand_gal_dec, triweight, h_opt)))/N)              # gamma  in the paragraph between Eq. 19 and Eq. 20 of Serra et al. 2011
        lam = (gamma/fq(cand_gal_ra, cand_gal_dec, cand_gal_ra, cand_gal_dec, triweight, h_opt))**0.5                             # lambda in the paragraph between Eq. 19 and Eq. 20 of Serra et al. 2011
        h = h_c*h_opt*lam                                                                                                         # local smoothing length
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
        den = fq(X, Y, cand_gal_ra, cand_gal_dec, triweight, h)                                 # estimated number density at each point (X, Y)
        max_dec_idx, max_ra_idx = np.unravel_index(np.argmax(den, axis=None), den.shape)        # index of RA grid and Dec grid where the number density is maximum
        ra_cl = ra_grid[max_ra_idx]                                                             # RA  of cluster center
        dec_cl = dec_grid[max_dec_idx]                                                          # Dec of cluster center

        return ra_cl, dec_cl, v_cl


def find_kappa(r_grid, v_grid, r_res, den, R_avg, v_var, grad_limit):
        
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

    fn = lambda kappa: S(kappa, r_grid, v_grid, r_res, den, R_avg, v_var, grad_limit)
    k_min = 0
    k_max = den.max()
    nstep = 101
    step_size = (k_max - k_min) / nstep
    
    k0 = k_min
    k1 = k0 + step_size
    S0 = fn(k0)
    S1 = fn(k1)
    
    i = 2
    decreasing = (S1 <= S0)
    while(i < nstep):
        k2 = k1 + step_size
        S2 = fn(k2)
        
        if (S2 > S1) & (decreasing):
            k = parabola_min(k0, k1, k2, S0, S1, S2)
            return k
        else:
            if (S2 <= S1): 
                decreasing = True
            k0 = k1
            k1 = k2
            S0 = S1
            S1 = S2
            i += 1

def S(kappa, r_grid, v_grid, r_res, den, R_avg, v_var, grad_limit):

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
    A, dA = calculate_A(kappa, den, r_grid, v_grid, grad_limit)                        # amplitude of the caustic lines for given kappa
    
    # phi is calculated by integrating fq within the caustic lines.
    phi = np.empty(r_res)                                                   # initialize phi as an empty array with the same size as the r_grid
    for i in range(phi.size):                                               # for each grid points in r_grid, calculate phi
        phi[i] = np.trapz(den[(v_grid < A[i]) & (v_grid > -A[i]), i], x = v_grid[(v_grid < A[i]) & (v_grid > -A[i])])          # 1D integration of of number density fixed at r_grid[i], and integration range is only within the caustic lines

    if np.trapz(phi[r_grid < R_avg], x = r_grid[r_grid < R_avg]) == 0:              # If the integration of phi along r_grid is 0, this causes divided by zero. Thus, we just return infinity to avoid warning/error.
        return np.inf

    v_esc_mean_squared = np.trapz((A[r_grid < R_avg]**2) * phi[r_grid < R_avg], x = r_grid[r_grid < R_avg]) / np.trapz(phi[r_grid < R_avg], x = r_grid[r_grid < R_avg])       # mean squared escape velocity
    
    v_mean_squared = v_var                                                   # mean squared velocity (calculated in advance from candidate member galxies)
    print("kappa = {:.6e}: v^2 = {:.6e}, v_esc^2 = {:.6e}, S = {:.6e}".format(kappa, np.sqrt(v_mean_squared), np.sqrt(v_esc_mean_squared), abs(v_esc_mean_squared - 4*v_mean_squared)))
    return abs(v_esc_mean_squared - 4*v_mean_squared)                       # return value: absolute value of the difference of mean sqaured escape velocity and mean squared velocity



def calculate_A(kappa, den, r_grid, v_grid, grad_limit):

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
    caustic amplitude and its uncertainty at each point in r_grid
    '''
    contours = skimage.measure.find_contours(den, kappa)                    # contour lines of number density for given level kappa

    v_step = v_grid[1] - v_grid[0]                                          # step size of v_grid

    A = np.full_like(r_grid, fill_value=max(v_grid))                        # amplitude of the caustic lines determined at each grid value of r_grid; initialize with the maximum absolute value of the v_grid
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

        '''
* Note on algorithm *
We take the minimum abs(v) value (for a given r) as the caustic amplitude.
There may be several isolated contour lines for a single level kappa.

If the contour is on only the positive or negative v half-plane
(i.e., the contour does not cross v=0), we just update A(r) with the minimum
value of abs(v) for given r. No subtleties here.

If the contour crosses v=0 line, we separate the case where the contour is below
v=0 and above v=0. 
For each case, we start from r=0 and follow the contour line in increasing r.
If A(r) is greater than the abs(v) for the current r, we update A(r) with 
abs(v).

These measures seemed necessary to be consistent with Caustic App.
        '''

        r_idx_0 = 0
        if np.sign(v_cont[0]*v_step + v_grid.min()) == np.sign(v_cont[-1]*v_step + v_grid.min()):   # the contour is only on one side of v=0
            for r_idx, v_idx in zip(r_cont_grid, v_cont_grid):              # for coordinate indices (r_idx, v_idx) on the contour line
                v = v_idx*v_step + v_grid.min()                             # v value on the v_grid corresponding to v_idx
                if check_sign and negative and (v >= 0):                    # 1) If we need to check the sign change of v (i.e., the contour is non-looping), 2) if the contour started from negative v, and 3) if the current v value is positive, then we have found where the contour line crosses v = 0. 
                    zero_idx = r_idx+1                                      # In this case, all values of A should be 0 from r_idx + 1.
                    negative = False                                        # And as the v value has became positive, set negative to False.
                
                A[r_idx] = min(A[r_idx], abs(v))                            # Caustic line amplitude is the minimum value of abs(v) for given r.
        
        elif negative:
            for idx in range(len(r_cont_grid)):
                r_idx = r_cont_grid[idx]
                v_idx = v_cont_grid[idx]
                v = v_idx*v_step + v_grid.min()                             # v value on the v_grid corresponding to v_idx
                if check_sign and negative and (v >= 0):                    # 1) If we need to check the sign change of v (i.e., the contour is non-looping), 2) if the contour started from negative v, and 3) if the current v value is positive, then we have found where the contour line crosses v = 0. 
                    zero_idx = r_idx+1                                      # In this case, all values of A should be 0 from r_idx + 1.
                    negative = False
                    r_idx_0 = r_idx                                         # r index where the contour crosses v=0
                    break                                                   # we now need to start from the upper contour
                    
                if negative:
                    A[r_idx] = min(A[r_idx], abs(v))                        # Caustic line amplitude is the minimum value of abs(v) for given r.
    
            for idx in range(len(r_cont_grid)-1, r_idx_0, -1):
                r_idx = r_cont_grid[idx]
                v_idx = v_cont_grid[idx]
                v = v_idx*v_step + v_grid.min()
                if ~negative:
                    A[r_idx] = min(A[r_idx], abs(v))
                

    if zero_idx != None:                                                    # zero_idx is initialized as None before the loop. Thus, if zero_idx is NOT None, then there is a point where the contour line crossed v = 0.
        A[zero_idx:] = 0                                                    # A(r) is set to 0 for r >= r_grid[zero_idx].

    A = grad_restrict(A, r_grid, grad_limit=grad_limit)           # restrict gradient of A
    dA = A*kappa/np.max(den, axis=0)
    
    return A, dA                                                            # return value: amplitude of caustic line and its uncertainty determined at grid points on r_grid

def grad_restrict(A, r, grad_limit, new_grad = 0.25):

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
            log_grad = r[i]/A[i] * (A[i+1]-A[i])/(r[i+1]-r[i])                                  # gradient of the current A
            if log_grad > grad_limit:
                A[i+1] = np.exp( np.log(A[i]) + new_grad*(np.log(r[i+1]) - np.log(r[i])) )      # new value of A[i+1]
            
    return A

def calculate_M(r_grid, A, dA, F_b):
        
    '''
    Calculates the enclosed mass profile and its uncertainty.

    Parameters
    --------------------------------------
    r_grid  : numpy ndarray,    grid along the r-axis
    A       : numpy ndarray,    caustic amplitude at given radius
    dA      : numpy ndarray,    uncertainty of the caustic amplitude
    F_b     : float,            filling factorr

    Returns
    --------------------------------------
    enclosed mass profile and its uncertainty at each point in r_grid
    '''

    M = np.zeros_like(r_grid)       # enclosed mass profile
    dM = np.zeros_like(r_grid)      # uncertainty of M

    # calcualte M
    for i in range(len(M)):
        M[i] = F_b*np.trapz(x=r_grid[0:i], y=A[0:i]**2)
    M *= ((u.km/u.s)**2*(u.Mpc)/const.G).to_value(u.M_sun)  # covert to solar mass

    # calculate dM
    for i in range(1, len(dM)):
        if (A[i] == 0):
            break
        for j in range(1, i):
            dm = M[j]-M[j-1]        # mass of the j-th shell
            dM[i] += 2*np.abs(dm*dA[j]/A[j])
    
    return M, dM

def membership(r, v, r_grid, A):

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

def velocity_dispersion(v, give_error=False, N_resamp=1000):
    '''
    Calculates the velocity dispersion of galaxies.
    If `give_error=True`, the function also calculates 1 sigma resampling error.
    In this case, the returned value is a 2-tuple (vdisp, vdisp_error) in units of (km/s, km/s).
    Otherwise, the returned value is simply a scalar vdisp in units of (km/s).
    If `give_error=False`, `N_resamp` is ignored.

    Parameters
    -----------------------------
    give_error  : bool, True if to calculate resampling error; default value False
    N_resamp    : int,  Number of resamplings to calculate error; default value 1000

    Returns
    -----------------------------
    Velocity dispersion of member galaxies in units of (km/s).
    If `give_error=True`, the returned value is 2-tuple (vdisp, vdisp_error).
    '''
    
    if give_error == False:
        return np.std(v, ddof=1)
    
    else:
        N = len(v)                                                         # number of members
        vdisp = np.std(v, ddof=1)                                     # velocity dispersion

        rng = np.random.default_rng(seed=0)                                             # random number generator with fixed seed 0
        rand_idx = rng.integers(low=0, high=N, endpoint=False, size=(N, N_resamp))      # random sampled indices
        vdisp_resamp = np.std(v[rand_idx], axis=0, ddof=1)            # resampled vdisp; shape=N_resamp

        vdisp_err = np.std(vdisp_resamp)                                                # resampling error

        return vdisp, vdisp_err

class CausticResults:
    def __init__(self, ra_gal, dec_gal, v_gal, r, v, member, ra_cl, dec_cl, v_cl, ra_cl_cand, dec_cl_cand, v_cl_cand, vdisp, vdisp_err, r_grid, v_grid, A, dA, M, dM, BT_mainbranch, BT_sigma, BT_cut_idx):
        self.ra_gal = ra_gal
        self.dec_gal = dec_gal
        self.v_gal = v_gal
        self.r = r
        self.v = v
        self.member = member
        self.ra_cl = ra_cl
        self.dec_cl = dec_cl
        self.v_cl = v_cl
        self.ra_cl_cand = ra_cl_cand
        self.dec_cl_cand = dec_cl_cand
        self.v_cl_cand = v_cl_cand
        self.vdisp=vdisp
        self.vdisp_err=vdisp_err
        self.r_grid = r_grid
        self.v_grid = v_grid
        self.A = A
        self.dA = dA
        self.M = M
        self.dM = dM
        self.BT_mainbranch = BT_mainbranch
        self.BT_sigma = BT_sigma
        self.BT_cut_idx = BT_cut_idx

    def create_member_list(self, output_fpath):
        N = len(self.ra_gal)
        np.savetxt(fname=output_fpath,
                   X=np.stack([self.ra_gal, self.dec_gal, self.v_gal, self.member.astype(int)]).T,
                   header=f"{N} {self.ra_cl:.6f} {self.dec_cl:.6f} {self.v_cl:.0f}",
                   comments='',
                   fmt="%9.6f\t%9.6f\t%7.0f\t%2d")