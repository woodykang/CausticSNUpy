import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
import astropy.coordinates
import astropy.units as u
import astropy.cosmology
import astropy.stats
from hierarchical_clustering import hier_clustering


def Caustics(fpath, v_lower, v_upper, r_max, H0 = 100, Om0 = 0.3, Ode0 = 0.7, Tcmb0 = 2.7, q = 25, r_res = 100, v_res = 100, BT_thr = "ALS"):
    
    '''
    Inputs
    ----------------
    fpath   : str,   file path of the txt file that contains the information of the galaxies. 
    v_lower : float, upper bound of line of sight velocities that member galaxies can have
    v_upper : float, lower bound of line of sight velocities that member galaxies can have
    rmax    : float, maximum radius in which the member galaxies lie
    H0      : float, Hubble constant in km/s/Mpc; default value 100
    Om0     : float, matter density parameter for the cosmological model; default value 0.3
    Ode0    : float, dark energy density paramter for the cosmological model; default value 0.7
    Tcmb0   : float, Temperature of the CMB at z=0; default value 2.7
    q       : float, scaling factor (see Section 4.1 of Diaferio 1999); default value 25
    r_res   : int,   resolution of the redshift diagram in the r-axis; default value 100
    v_res   : int,   resolution of the redshift diagram in the v-axis; default value 100
    BT_thr  : str,   Binary Tree treshold, should be either "AD" or "ALS" (currently only ALS is supported); default value "ALS"

    Outputs
    ----------------
    r_grid  : numpy ndarray, r-axis grid of the redshift diagram
    v_grid  : numpy ndarray, v-axis grid of the redshift diagram
    A       : numpy ndarray, amplitude of the caustic lines evaluated at r_grid
    den     : numpy ndarray, number density on redshift diagram evaluted at r_grid and v_grid
    r       : numpy ndarray, projected distance from cluster center to each galaxy (Mpc)
    v       : numpy ndarray, relative l.o.s. velocity with regard to the cluster center (km/s)
    member  : numpy ndarray, 0 if the galaxy is an interloper and 1 if the galaxy is a member
    '''
    
    # constants
    c     = 299792.458     # speed of light in km/s
    
    # unpack data
    print("Unpacking data.")
    cluster_data = np.loadtxt(fpath, max_rows = 1)  # In the first row, the format is N, cl_ra, cl_dec, cl_v
                                                    # where N is number of galaxies observed, and
                                                    # cl_ra, cl_dec, cl_z are RA (deg), DEC (deg), los velocity (km/s) of the cluster
    cl_ra, cl_dec, cl_v = cluster_data[1:]
    gal_ra, gal_dec, gal_v = np.loadtxt(fpath, skiprows=1, unpack=True)     # RA (deg), Dec (deg), l.o.s velocity (km/s)
    
    
    # calculate projected distance and radial velocity
    LCDM = astropy.cosmology.LambdaCDM(H0, Om0, Ode0, Tcmb0)    #Lambda CDM model with the given parameters
    
    cl_z = cl_v/c                                               # redshift of the cluster center
    d_A = LCDM.angular_diameter_distance(cl_z)                  # angular diameter distance to the cluster center
    
    angle = astropy.coordinates.angular_separation(cl_ra*np.pi/180, cl_dec*np.pi/180, gal_ra*np.pi/180, gal_dec*np.pi/180)      # angular separation of galxay and cluster center
    r = (angle*d_A).to(u.Mpc, equivalencies = u.dimensionless_angles()).value                                                   # projected distance from cluster center to each galaxy (in Mpc)
    v = (gal_v - cl_v)/(1+cl_z)                 # relative l.o.s velocity with regard to cluster center

    # apply cutoffs given by input
    cutoff_idx = (gal_v > v_lower) & (gal_v < v_upper) & (r < r_max)
    gal_ra  = gal_ra[ cutoff_idx]
    gal_dec = gal_dec[cutoff_idx]
    gal_v   = gal_v[  cutoff_idx]
    r = r[cutoff_idx]
    v = v[cutoff_idx]
    
    v_min = v_lower - cl_v      # lower bound of v
    v_max = v_upper - cl_v      # upper bound of v

    print("Number of galaxies in vel and r_max limit : {}".format(r.size))

    # shortlist candidate members using hierarchical clustering
    cand_mem_idx = hier_clustering(gal_ra, gal_dec, gal_v, threshold=BT_thr)            # indices of candidate members

    print("Number of candidate members : {}".format(len(cand_mem_idx)))

    vvar = np.var(v[cand_mem_idx], ddof=1)      # variance of v calculated from candidate members; later to be used for function S(k)
    R = np.average(r[cand_mem_idx])             # average projected distance from the center of the cluster to candidate member galaxies; later to be used for function S(k)
    #vvar = 606.156616**2
    #R = 0.950363


    print("Velocity Dispersion : {} km/s".format(np.sqrt(vvar)))
    print("Mean distance       : {} Mpc".format(R))

    print("Data unpacked.")
    print("")
    
    # estimate number density in the redshift diagram using adaptive kernel density estimation
    print("Estimating number density in phase space.")
    
    x_data = r*H0                               # rescaled data points in r-axis (in units of km/s)
    y_data = v/q                                # rescaled data points in v-axis (in units of km/s)
    
    r_min = 0                                   # minimum of r should be, obviously, 0
    r_grid = np.linspace(r_min, r_max, r_res)   # grid of r-axis on the redshift diagram
    v_grid = np.linspace(v_min, v_max, v_res)   # grid of v-axis on the redshift diagram

    x_grid = r_grid*H0                          # grid of rescaled r-axis (in units of km/s)
    y_grid = v_grid/q                           # grid of rescaled v-axis (in units of km/s)

    f = Diaferio_density(x_data, y_data, x_grid, y_grid)        # number density on redshift diagram
    
    print("Number density estimation done.")
    print("")
    
    r_min = 0                                   # minimum of r should be, obviously, 0
    r_grid = np.linspace(r_min, r_max, r_res)   # grid of r-axis on the redshift diagram
    v_grid = np.linspace(v_min, v_max, v_res)   # grid of v-axis on the redshift diagram

    x_grid = r_grid*H0                          # grid of rescaled r-axis (in units of km/s)
    y_grid = v_grid/q                           # grid of rescaled v-axis (in units of km/s)

    X, Y = np.meshgrid(x_grid, y_grid)          # mesh grid
    den = f(X, Y)                               # number density estimated at each point X, Y

    # minimize S(k) to get optimal kappa value
    print("Minimizing S(k) to find kappa.")

    a = den.min()
    b = den.max()
    fn = lambda kappa: S(kappa, r, v, r_grid, v_grid, den, R, vvar)             # lambda function is used to fix other parameters except for kappa; thus, fn is only a function of kappa
    kappa = minimize_fn(fn, a, b, positive = True, search_all = True)           # kappa that minimizes fn (= S(k))
      
    print("kappa found. kappa =  {}, S(k) = {}.".format(kappa, fn(kappa)))
    print("")
   
    # calculate A(r) with the minimized kappa
    print("Drawing caustic lines.")
    A = calculate_A(kappa, den, r_grid, v_grid)                                 # amplitude of the caustic lines drawn from den at level kappa
    
    '''
    # plot results
    plt.plot(r, v, ".", c = "y", alpha = 0.5)

    dmax = np.max(den)
    dmin = np.min(den)
    conf = plt.contourf(r_grid, v_grid, den, levels = np.linspace(dmin,dmax,100), cmap = "coolwarm", alpha=1)    # filled contour
    con = plt.contour(r_grid, v_grid, den, levels = (kappa,))

    plt.plot(r_grid,  A, color = "orange", label = "caustic")
    plt.plot(r_grid, -A, color = "orange")
    plt.plot(r_grid, A, "o", color = 'orange')

    plt.xlim(r_min, 3)
    plt.ylim(-2000, 2000)

    plt.title("Redshift Diagram")
    plt.xlabel("Projected Distance (Mpc/h)")
    plt.ylabel("$(v_{gal}-v_{cl})/(1+z)$ (km/s)")


    contours = skimage.measure.find_contours(den, kappa)
    for contour in contours:
        r_cont = contour[:,1]
        v_cont = contour[:,0]
        int_idx = (r_cont == r_cont.astype(int))
        plt.plot(r_grid[r_cont[int_idx].astype(int)], v_grid[v_cont[int_idx].astype(int)], c = 'green')
        #plt.plot(contour[:,1]*(r_grid[1]-r_grid[0]), contour[:,0]*(v_grid[1]-v_grid[0]) + v_grid.min(), c = 'green')
    plt.show()
    # end of plot results
    '''
    # determine membership
    member = membership(r, v, r_grid, A)    # array of 0 and 1; 0 for interlopers and 1 for members

    return r_grid, v_grid, A, den, r, v, member

def Diaferio_density(x_data, y_data, x_grid, y_grid):
    
    '''
    Description
    -----------------------------------
    Estimating number density on redshift diagram using adaptive kernel density estimation.

    Inputs
    -----------------------------------
    x_data  : numpy ndarray, 1D array containing rescaled r values of galaxies
    y_data  : numpy ndarray, 1D array containing rescaled v values of galaxies

    Output
    ----------------------------------
    funtion that returns the estimated number density on redshift diagram at given point (x, y). 
    '''
    # data points are mirrored to negative r (see the last two sentences from the second paragraph of Section 4.3, Serra et al. 2011)
    x_data_mirrored = np.concatenate((x_data, -x_data))
    y_data_mirrored = np.concatenate((y_data,  y_data))

    N = x_data.size                                                                                                                     # number of data points
    h_opt = 6.24/(N**(1/6)) * np.sqrt((np.std(x_data, ddof=1)**2 + np.std(y_data,ddof=1)**2)/2)                                         # eq. 20 from Serra et al. 2011
    print("h_opt : {}".format(h_opt))
    
    gamma = 10**(np.sum(np.log10(fq(x_data_mirrored, y_data_mirrored, x_data_mirrored, y_data_mirrored, triweight, h_opt)))/(2*N))      # gamma  defined in Diaferio 1999, between eq. 17 and eq. 18; here, the term is divided by 2*N because N is the number of original (i.e. un-mirrored) data points
    lam = (gamma/fq(x_data_mirrored, y_data_mirrored, x_data_mirrored, y_data_mirrored, triweight, h_opt))**0.5                         # lambda defined in Diaferio 1999, between eq. 17 and eq. 18

    fn = lambda h_c: h_cost_function(h_c, h_opt, lam, x_data_mirrored, y_data_mirrored, x_grid, y_grid)                                                 # M_0 function defined in eq. 18 from Diaferio 1999

    print("Calculating h_c.")
    h_c = iterative_minimize(fn, init = 0.005, step = 0.01, max = 10, print_proc=True)                                                  # imiated the minimization method used in CausticApp v1.6
    print("h_c : {}".format(h_c))
    
    h = h_c * h_opt * lam                                                                                                               # final h_i (local smoothing length); size of h_i is same as x_data and y_data

    return lambda x, y: fq(x, y, x_data_mirrored, y_data_mirrored, triweight, h)                                                        # return value: function that returns the number density on redshift diagram at given point (x, y). 

def h_cost_function(h_c, h_opt, lam, x_data, y_data, x_grid, y_grid):
    
    '''
    Description
    -----------------------------------
    Cost function which h_c should minimize (eq. 18 from Diaferio 1999).

    Inputs
    -----------------------------------
    h_c     : float,            smoothing factor h_c
    h_opt   : float,            optimal smoothing length (eq. 20 from Serra et al. 2011)
    lam     : numpy ndarray,    local smoothing length; same size as x_data and y_data
    x_data  : numpy ndarray,    1D array containing rescaled r values of galaxies
    y_data  : numpy ndarray,    1D array containing rescaled v values of galaxies

    Output
    ----------------------------------
    Value of cost function evaluated for given h_c     
    '''

    h = h_c * h_opt * lam       # local smoothing length
    N = x_data.size             # number of data points

    # calculating the first term
    ## set up grids for numerical integration; x_grid and y_grid here is different from those used in the main function Caustics().
    ## Note that the value of triweight funtion is 0 outside (x/h)**1 + (y/h)**1 = 1
    #x_min = 0 #np.min(x_data - h)      # minimum value in the x_grid
    #x_max = np.max(x_data + h)      # maximum value in the x_grid
    #y_min = np.min(y_data - h)      # minimum value in the y_grid
    #y_max = np.max(y_data + h)      # maximum value in the y_grid
    
    #x_res = 100                     # resolution of x_grid
    #y_res = 100                     # resolution of y_grid

    #x_grid = np.linspace(x_min, x_max, x_res)   # grid along rescaled r-axis (x-axis) used for numerical integration
    #y_grid = np.linspace(y_min, y_max, y_res)   # grid along rescaled v-axis (y-axis) used for numerical integration

    X, Y = np.meshgrid(x_grid, y_grid)          # mesh grid
    
    f_squared = fq(X, Y, x_data, y_data, triweight, h)**2           # squared value of fq

    term_1 = np.trapz(np.trapz(f_squared, x = y_grid, axis = 0), x = x_grid)       # first term of M_0 is the integration of fq squared
    
    # calculating the second term (refer to eq. 3.37 and Section 5.3.4 of Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London)
    x_pairs = np.subtract.outer(x_data, x_data)             # 2D array of size (N, N); element (i, j) is x_data[i]-x_data[j] i.e. pair-wise subtraction of x_data and x_data
    y_pairs = np.subtract.outer(y_data, y_data)             # 2D array of size (N, N); element (i, j) is y_data[i]-y_data[j] i.e. pair-wise subtraction of y_data and y_data
    a = np.sum(triweight(x_pairs/h, y_pairs/h) / (h**2))    # sum of fq evaluated at all data points (x, y)
    b = triweight(0, 0) * np.sum(1/(h**2))
    term_2 = 2/(N*(N - 1)) * (a - b)

    return term_1 - term_2


def M_0(h_c, h_opt, lam, x_data, y_data):
    h = h_c * h_opt * lam       # local smoothing length
    N = x_data.size             # number of data points

    # calculating the first term
    ## set up grids for numerical integration; x_grid and y_grid here is different from those used in the main function Caustics().
    ## Note that the value of triweight funtion is 0 outside (x/h)**1 + (y/h)**1 = 1
    x_min = np.min(x_data - h)      # minimum value in the x_grid
    x_max = np.max(x_data + h)      # maximum value in the x_grid
    y_min = np.min(y_data - h)      # minimum value in the y_grid
    y_max = np.max(y_data + h)      # maximum value in the y_grid
    
    x_res = 200                     # resolution of x_grid
    y_res = 100                     # resolution of y_grid

    x_grid = np.linspace(x_min, x_max, x_res)   # grid along rescaled r-axis (x-axis) used for numerical integration
    y_grid = np.linspace(y_min, y_max, y_res)   # grid along rescaled v-axis (y-axis) used for numerical integration

    X, Y = np.meshgrid(x_grid, y_grid)          # mesh grid
    
    f_squared = fq(X, Y, x_data, y_data, triweight, h)**2           # squared value of fq

    term_1 = np.trapz(np.trapz(f_squared, x = y_grid, axis = 0), x = x_grid)       # first term of M_0 is the integration of fq squared

    term_2 = 2/(N*(N-1)) * np.sum([[triweight((x_data[i] - x_data[j])/h[j], (y_data[i] - y_data[j])/h[j]) / (h[j]**2) for j in list(range(N))[:i] + list(range(N))[i+1:]] for i in range(N)])

    return term_1 - term_2

def triweight(x, y):
    '''
    Description
    -----------------------------------
    2D triwegith kernel

    Inputs
    -----------------------------------
    x   : float, x coordinate 
    y   : float, y coordinate

    Output
    ----------------------------------
    Value of triweight kernel evaulated at (x, y).     
    '''

    t = np.sqrt(x**2 + y**2)
    return 4/np.pi * ((1-t**2)**3) * (t < 1)        # return 4/pi * (1-t**2)**3 if t<1; otherwise 0

def fq(x, y, x_data, y_data, K, h):
    '''
    Description
    --------------------------------------
    Estimated number density function, using kernel density estimation.
    
    Inputs
    ---------------------------------------
    x       : float or numpy ndarray,   x-axis value(s) of the point(s) where the density is to be calculated
    y       : float or numpy ndarray,   y-axis value(s) of the point(s) where the density is to be calculated
    x_data  : numpy ndarray,            x-axis value of the observed data points
    y_data  : numpy ndarray,            y-axis value of the observed data points
    K       : function,                 kernel function
    h       : float or numpy ndarray,   bandwidth of the kernel
    
    Output
    ---------------------------------------
    Estimated density value at the given point (x, y).
    '''

    N = x_data.size                             # number of data points
    
    x_pairs = np.subtract.outer(x, x_data)      # 2D array of size (M, N) where M is the length of x; element (i, j) is x[i]-x_data[j] i.e. pair-wise subtraction of x and x_data
    y_pairs = np.subtract.outer(y, y_data)      # 2D array of size (M, N) where M is the length of y; element (i, j) is y[i]-y_data[j] i.e. pair-wise subtraction of y and y_data
    
    return np.sum(K(x_pairs/h, y_pairs/h)/(h**2), axis = -1) / N

def S(kappa, r, v, r_grid, v_grid, den, R, vvar):
    A = calculate_A(kappa, den, r_grid, v_grid)
    
    r_min = r_grid.min()
    r_max = r_grid.max()
    r_res = r_grid.size
    
    v_min = v_grid.min()
    v_max = v_grid.max()
    v_res = v_grid.size
    
    
    # phi is calculated by integrating f_q within the caustic lines.
    phi = np.empty(r_res)
    for i in range(phi.size):
        phi[i] = np.trapz(den[(v_grid < A[i]) & (v_grid > -A[i]), i], x = v_grid[(v_grid < A[i]) & (v_grid > -A[i])])

    if np.trapz(phi[r_grid < R], x = r_grid[r_grid < R]) == 0:
        return np.inf

    v_esc_mean_squared = np.trapz((A[r_grid < R]**2) * phi[r_grid < R], x = r_grid[r_grid < R]) / np.trapz(phi[r_grid < R], x = r_grid[r_grid < R])
    
    v_mean_squared = vvar              # In Diaferio 1999 and Serra et al. 2011, the mean of the squared velocity is independent of kappa.
    return abs(v_esc_mean_squared - 4*v_mean_squared)
    


def calculate_A(kappa, den, r_grid, v_grid):
    contours = skimage.measure.find_contours(den, kappa)
    r_step = r_grid[1] - r_grid[0]
    v_step = v_grid[1] - v_grid[0]

    A = np.full(r_grid.size, np.inf)
    zero_idx = None
    for contour in contours:
        r_cont = contour[:,1]
        v_cont = contour[:,0]

        check_sign = False
        negative = False
        if not (r_cont[0] == r_cont[-1] and v_cont[0] == v_cont[-1]):
            check_sign = True

        if v_cont[0]*v_step + v_grid.min() < 0:
            negative = True
        else:
            negative = False
        
        int_idx = (r_cont == r_cont.astype(int))
        r_cont_grid = r_cont[int_idx].astype(int)
        v_cont_grid = v_cont[int_idx]
        for r, v in zip(r_cont_grid, v_cont_grid):
            v = v*v_step + v_grid.min()
            if check_sign and negative and (v >= 0):
                zero_idx = r+1
                negative = False
            A[r] = min(A[r], abs(v))

    if zero_idx != None:
        A[zero_idx:] = 0
    A[np.isinf(A)] = 0
    A = grad_restrict(A, r_grid)
    
    return A

def minimize_fn(fn, a, b, positive = False, it = 0, search_all = False):
    print("search range: {} ~ {}".format(a, b))
    
    it += 1
    
    if positive:
        a = max(a, 0)
        b = max(b, 0)
        if (a == 0) and (b == 0):
            raise Exception("Cannot optimize the function. Either reset the search range, or check if the function is optimizable.")
    
    if (it == 1) & (search_all):
        search_range = np.logspace(np.log10(a + abs(b-a)*1e-6), np.log10(b), 100)
        min_idx = 0
        min_val = fn(search_range[0])
        for i in range(search_range.size):
            x = search_range[i]
            val = fn(x)
            if val < min_val:
                min_idx = i
                min_val = val
        
        print("min_idx : {}, min_x : {}, min_val : {}".format(min_idx, search_range[min_idx], min_val))
        #return search_range[min_idx]
        a = max(search_range[min_idx - 2] if min_idx > 1 else search_range[min_idx - 1], a)
        b = min(search_range[min_idx + 2] if min_idx < search_range.size-2 else search_range[min_idx + 1], b)
        print("search range: {} ~ {}".format(a, b))

    a_init = a
    b_init = b
    
    R = (np.sqrt(5)-1)*0.5        # golden ratio
    
    TOL = 1e-6 * abs(b - a)
    if TOL == 0:
        TOL = 1e-5
    
    error = TOL + 1
    
    x1 = b - R*(b - a)
    x2 = a + R*(b - a)
    
    f1 = fn(x1)
    f2 = fn(x2)
    
    while error > TOL:
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
        
        print("iteration : {}".format(it), end = '\r')
        
        it += 1
        if it > 100:
            raise Exception("Cannot optimize the function. Either reset the search range, or check if the function is optimizable.")
    
    print("iteration : {}".format(it))
    if abs(a_init - a) < TOL:
        a = a_init - (b_init - a_init)
        b = a_init
        
        print("Shifting search range to ({}, {})".format(a, b))
        return minimize_fn(fn, a, b, positive, it)
        
    elif abs(b_init - b) < TOL:
        a = b_init
        b = b_init + (b_init - a_init)
        
        print("Shifting search range to ({}, {})".format(a, b))
        return minimize_fn(fn, a, b, positive, it)
    
    print("value : {} / fn : {}".format(a, f1))
    return a

def iterative_minimize(fn, init, step, max, print_proc = False):
    x0 = init
    f0 = fn(x0)

    x1 = init + step
    f1 = fn(x1)

    x2 = init + 2*step
    f2 = fn(x2)
    if print_proc:
        print("h_c = {:.6f} : fn = {:.6e}".format(x0, f0))
        print("h_c = {:.6f} : fn = {:.6e}".format(x1, f1))
        while(x1 < max):
            print("h_c = {:.6f} : fn = {:.6e}".format(x2, f2))
            if (f0 > f1) & (f1 < f2):
                print("Parabolic minimize.")
                x = x1 - 0.5 * ((x1 - x0)**2 * (f1 - f2) - (x1 - x2)**2 * (f1 - f0))/( (x1 - x0)*(f1 - f2) - (x1 - x2)*(f1 - f0))
                return x
            x0 += step
            x1 += step
            x2 += step

            f0 = f1
            f1 = f2
            f2 = fn(x2)
    
    else:
        while(x1 < max):
            if (f0 > f1) & (f1 < f2):
                print("Parabolic minimize.")
                x = x1 - 0.5 * ((x1 - x0)**2 * (f1 - f2) - (x1 - x2)**2 * (f1 - f0))/( (x1 - x0)*(f1 - f2) - (x1 - x2)*(f1 - f0))
                return x
            x0 += step
            x1 += step
            x2 += step

            f0 = f1
            f1 = f2
            f2 = fn(x2)
    
    if x1 == max:
        raise Exception("Unable to minimize function.")

    return False


def grad_restrict(A, r, grad_limit = 2, new_grad = 0.25):
    for i in range(A.size-1):
        if (A[i] <= 0) or (A[i+1] <= 0) or (r[i] == 0):
            continue
        
        #if A[i+1] <= 0:
        #    A[i+1] = np.exp( np.log(A[i]) - grad_limit*(np.log(r[i+1]) - np.log(r[i])) )
        
        else:
            log_grad = (np.log(A[i+1])-np.log(A[i])) / (np.log(r[i+1]) - np.log(r[i]))
            if log_grad > grad_limit:
                A[i+1] = np.exp( np.log(A[i]) + new_grad*(np.log(r[i+1]) - np.log(r[i])) )
            
    return A

def membership(r, v, r_grid, A):
    A_at_r = np.interp(r, r_grid, A)
    return abs(v) < A_at_r