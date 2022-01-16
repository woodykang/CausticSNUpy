import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from matplotlib import cm
import astropy.coordinates
import astropy.cosmology


def Caustics(fpath, r200, v_lower, v_upper, r_max = None, r_res = 250, v_res = 250, method = "Gifford"):
    '''
    Inputs
    ----------------
    fpath    : str,   file path of the txt file that contains the information of the galaxies. 
    r200     : float, virial radius of the cluster
    v_lower  : float, upper bound of line of sight velocities that member galaxies can have
    v_upper  : float, lower bound of line of sight velocities that member galaxies can have
    rmax     : float, maximum radius in which the member galaxies lie
    method   : str,   method of kernel density estimation
    '''
    
    # constants
    c     = 299792.458      # speed of light in km/s
    H     = 100             # Hubble constant in km/s/Mpc
    Om0   = 0.3             # matter density parameter for the cosmological model
    Ode0  = 0.7             # dark energy density paramter for the cosmological model
    Tcmb0 = 2.7             # Temperature of the CMB at z=0
    q     = 25              # scaling factor ratio of v to r
    
    # unpack data
    print("Unpacking data.")
    cluster_data = np.loadtxt(fpath, max_rows = 1)  # In the first row, the format is N, cl_ra, cl_dec, cl_v
                                                    # where N is number of galaxies observed, and
                                                    # cl_ra, cl_dec, cl_z are RA (deg), DEC (deg), los velocity (km/s) of the cluster
    cl_ra, cl_dec, cl_v = cluster_data[1:]
    gal_ra, gal_dec, gal_v = np.loadtxt(fpath, skiprows=1, unpack=True)     # RA, Dec in degrees, los velocity in km/s
    
    
    # calculate projected distance and radial velocity
    LCDM = astropy.cosmology.LambdaCDM(H, Om0, Ode0, Tcmb0)     #Lambda CDM model with the given parameters
    
    cl_z = cl_v/c   # redshift of the cluster center
    R = float(LCDM.angular_diameter_distance(cl_z)/astropy.units.Mpc)   # angular diameter distance to the cluster center
    
    angle = astropy.coordinates.angular_separation(cl_ra*np.pi/180, cl_dec*np.pi/180, gal_ra*np.pi/180, gal_dec*np.pi/180)      #angular separation of galxay and cluster center
    r = angle*R                                 # projected distance from cluster center to each galaxies
    v = (gal_v - cl_v)/(1+cl_z)                 # relative los velocity with regard to cluster center
    
    # apply cutoffs given by input
    if r_max == None:
        r_max = r200*5
    
    cutoff_idx = (gal_v > v_lower) & (gal_v < v_upper) & (r < r_max)
    gal_ra  = gal_ra[ cutoff_idx]
    gal_dec = gal_dec[cutoff_idx]
    gal_v   = gal_v[  cutoff_idx]
    r = r[cutoff_idx]
    v = v[cutoff_idx]
    
    v_min = v_lower - cl_v      # lower bound of v
    v_max = v_upper - cl_v      # upper bound of v
    
    print("Data unpacked.")
    print("Cluster center coordinates: RA {} deg,  Dec {} deg".format(cl_ra, cl_dec))
    print("Cluster center radial velocity : {} km/s".format(cl_v))
    print("")
    
    # estimate number density in the redshift diagram
    print("Estimating number density in phase space via " + method + " method.")
    
    x_data = r*H
    y_data = v/q
    
    # mirror the data for boundary correction
    x_data_mirrored = np.concatenate((x_data, -x_data))
    y_data_mirrored = np.concatenate((y_data,  y_data))
    
    if method == "Gifford":
        f = Gifford_density(x_data_mirrored, y_data_mirrored)
    elif method == "AdaptiveGifford":
        f = adaptive_Gifford_density(x_data_mirrored, y_data_mirrored)
    elif method == "Diaferio":
        f = Diaferio_density(x_data_mirrored, y_data_mirrored)
    else:
        raise ValueError(method + "is not a valid method.")
    
    print("Number density estimation done.")
    print("")
    
    # calculate number density for a mesh grid
    #r_res = 250    # r-axis resolution for the mesh grid
    #v_res = 250    # v-axis resolution for the mesh grid
    
    # undo mirror
    r_min = 0
    
    r_grid = np.linspace(r_min, r_max, r_res)
    v_grid = np.linspace(v_min, v_max, v_res)
    
    x_grid = r_grid*H
    y_grid = v_grid/q
    
    X, Y = np.meshgrid(x_grid, y_grid)
    den = f(X, Y)
    

    # minimize S(k) to get optimal kappa value
    print("Minimizing S(k) to find kappa.")
    kappa_guess = np.average(den)
    #kappa_guess = (den.max() + den.min())/2
    a = kappa_guess*0.5
    b = kappa_guess*2.0
    #a = den.min()
    #b = den.max()
    fn = lambda kappa: S(kappa, r, v, r_grid, v_grid, den, r200)
    kappa = minimize_fn(fn, a, b, positive = True, search_all = True)

    
    
    print("kappa found. kappa =  {}, S(k) = {}.".format(kappa, fn(kappa)))
    print("")
   
    # calculate A(r) with the minimized kappa
    print("Drawing caustic lines.")
    A = calculate_A(kappa, den, r_grid, v_grid)
    
    
    plt.plot(r, v, ".", c = "y", alpha = 0.5)
    cmap = plt.cm.gist_heat
    dmax = np.max(den)
    dmin = np.min(den)
    conf = plt.contourf(r_grid, v_grid, den, levels = np.linspace(dmin,dmax,100),cmap = "coolwarm", alpha=1)    # filled contour
    plt.colorbar(conf)

    con = plt.contour(r_grid, v_grid, den, levels = (kappa,))

    plt.plot(r_grid,  A, color = "orange", label = "caustic")
    plt.plot(r_grid, -A, color = "orange")


    plt.xlim(r_min, r_max)
    plt.ylim(-2000, 2000)

    plt.title("Contour of # Density (Gaussian Kernel)")
    plt.xlabel("Projected Distance (Mpc/h)")
    plt.ylabel("$(v_{gal}-v_{cl})/(1+z)$ (km/s)")

    plt.show()
    
    return r_grid, v_grid, A, den

def Gifford_density(x_data, y_data):
    '''
    Estimates the number density function in the redshift diagram (phase space).
    This function uses method described in Gifford et al. 2013. 
    Basically, it uses standard kernel density estimation via Gaussian kernel with different bandwidths for r-axis and v-axis.
    The bandwidths are proportional to the standard deviations along each axis.
    
    Inputs
    ---------------------------------------------
    x_data: array-like, scaled data points along the r-axis
    y_data: array-like, scaled data points along the v-axis
    
    Note: The scaling is done by multiplying r by H, and dividing v by q.
    
    Output
    ---------------------------------------------
    Function that returns the number density at a given point x, y.
    x and y are r and v scaled, just as the input arguments.
    
    '''
    
    
    N = x_data.size                            # number of data points
    h_x = (4/(3*N))**0.2 * np.std(x_data)      # bandwidth along the x-axis (scaled r-axis)
    h_y = (4/(3*N))**0.2 * np.std(y_data)      # bandwidth along the y-axis (scaled v-axis)
    h = np.asarray([h_x, h_y])                 # bandwidth as an array
    
    return lambda x, y: fq(x, y, x_data, y_data, gauss2d, h)

def adaptive_Gifford_density(x_data, y_data):
    
    '''
    Estimates the number density function in the redshift diagram (phase space).
    This function is based on the method described in Gifford et al. 2013.
    However, instead of using the standard kernel denstiy, the adaptive kernel density estimation is applied.
    Thus, a local smoothing factor lambda is multiplied to the global bandwidth h.
    The changes made to employ the adaptive method are from Diaferio 1999.
    
    Inputs
    ---------------------------------------------
    x_data: array-like, scaled data points along the r-axis
    y_data: array-like, scaled data points along the v-axis
    
    Note: The scaling is done by multiplying r by H, and dividing v by q.
    
    Output
    ---------------------------------------------
    Function that returns the number density at a given point x, y.
    x and y are r and v scaled, just as the input arguments.
    
    '''
    
    N = x_data.size                         # number of data points
    h_x = (4/(3*N))**0.2 * np.std(x_data)   # global bandwidth along the x-axis (scaled r-axis)
    h_y = (4/(3*N))**0.2 * np.std(y_data)   # global bandwidth along the y-axis (scaled v-axis)

    gamma = 10**(np.sum(np.log10(fq(x_data, y_data, x_data, y_data, gauss2d, (h_x, h_y)))/N))
    lam = (gamma/fq(x_data, y_data, x_data, y_data, gauss2d, (h_x, h_y)))**0.5        # local smoothing factor
    #gamma = 10**(np.sum(np.log10(fq(x_data, y_data, x_data, y_data, gauss2d, (1, 1)))/N))
    #lam = gamma/fq(x_data, y_data, x_data, y_data, gauss2d, (1, 1))        # local smoothing factor
    h = np.vstack((h_x*lam, h_y*lam))                                          # bandwidth as an array
    
    return lambda x, y: fq(x, y, x_data, y_data, gauss2d, h)

def Diaferio_density(x_data, y_data):
    N = x_data.size
    h_opt = 6.24/(N**(1/6))*np.sqrt((np.std(x_data)**2 + np.std(y_data)**2)/2)
    
    gamma = 10**(np.sum(np.log10(fq(x_data, y_data, x_data, y_data, triweight, 1)))/N)
    #gamma = np.prod(fq(x_data, y_data, x_data, y_data, triweight, 1))**(1/N)
    lam = (gamma/fq(x_data, y_data, x_data, y_data, triweight, h_opt))**0.5
    
    fn = lambda h_c: h_cost_function(h_c, h_opt, lam, x_data, y_data)
    
    a = 1e-5
    b = 2
    print("Calculating h_c.")
    h_c = minimize_fn(fn, a, b, positive = True, search_all = True)
    print("h_c calculation finished. h_c = {}".format(h_c))
    h = h_c * h_opt * lam
    
    
    #h = h_opt * lam
    return lambda x, y: fq(x, y, x_data, y_data, triweight, h)

def h_cost_function(h_c, h_opt, lam, x_data, y_data):
    h = h_c * h_opt * lam
    N = x_data.size

    # calculating the first term
    x_min = np.min(x_data - h)
    x_max = np.max(x_data + h)
    y_min = np.min(y_data - h)
    y_max = np.max(y_data + h)
    
    x_res = 100
    y_res = 100
    x_grid = np.linspace(x_min, x_max, x_res)
    y_grid = np.linspace(y_min, y_max, y_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    f_squared = fq(X, Y, x_data, y_data, triweight, h)**2
    term_1 = np.trapz(np.trapz(f_squared, dx = x_grid[1] - x_grid[0], axis = 0), dx = y_grid[1] - y_grid[0])
    
    # calculating the second term
    x_pairs = np.subtract.outer(x_data, x_data)
    y_pairs = np.subtract.outer(y_data, y_data)
    a = np.sum(triweight(x_pairs, y_pairs, h) / (h[:, np.newaxis]**2))
    b = np.sum(triweight(0, 0, h) / (h[:, np.newaxis])**2)
    term_2 = 2/(N*(N - 1)) * (a - b)
    
    return term_1 - term_2

def triweight(x, y, h):
    d = np.sqrt(x**2 + y**2)
    try:
        t = (d/h[:, np.newaxis])
    except:
        t = d/h
    return 4/np.pi * ((1-t**2)**3) * (t < 1)

def gauss2d(x, y, std):
    '''
    2D Gaussian function.
    
    Inputs
    ---------------------------------
    x:    float, x-axis value of the point
    y:    float, y-axis value of the point
    std:  float or array-like, standard deviation of the Gaussian function. If std is given as array-like, then the first element and second element are used as the standard deviation of the x-axis and y-axis, respectively.
    
    Outputs
    ---------------------------------
    Function value at given point x, y.
    '''
    try:
        xstd, ystd = std
    except:
        xstd = std
        ystd = std
    
    return np.exp(-((x/xstd)**2 + (y/ystd)**2))

def fq(x, y, x_data, y_data, K, h):
    '''
    Estimated number density function, using kernel density estimation.
    
    Inputs
    ---------------------------------------
    x:        float, x-axis value of the point where the density is to be calculated
    y:        float, y-axis value of the point where the density is to be calculated
    x_data:   array-like, x-axis value of the observed data points
    y_data:   array-like, y-axis value of the observed data points
    K:        function, kernel function
    h:        float or array-like, bandwidth of the kernel.
    
    Output
    ---------------------------------------
    estimated density value at the given point x, y.
    '''
    
    
    N = x_data.size
    
    x_pairs = np.subtract.outer(x, x_data)
    y_pairs = np.subtract.outer(y, y_data)
    
           
    if np.size(h) == 1:
        s = np.sum(K(x_pairs, y_pairs, h)/(h**2), axis=-1)
    #elif len(h) > 2:
    #    s = np.sum(K(x_pairs, y_pairs, h)/(h[np.newaxis, np.newaxis:]**2), axis=-1)
    elif len(h) == 2:
        s = np.sum(K(x_pairs, y_pairs, h)/(h[0]**2 + h[1]**2), axis=-1)
    else:
        s = np.sum(K(x_pairs, y_pairs, h)/(h**2), axis=-1)
        
    return s/N

def coor2idx(x, xmin, xmax, nstep):       # nstep: number of steps
    '''
    Calculates the corresponding index of a value in a list.
    
    Inputs
    ------------------------------------
    x:      float, value to be converted to index
    xmin:   float, minimum value of the list
    xmax:   float, maximum value of the list
    nsteps: int, size of the array
    
    Output
    --------------------------------------
    idx: int, index corresponding to x in the list.
    '''
    
    if np.any(x > xmax):
        raise ValueError("x cannot be greater than xmax")
    if np.any(x < xmin):
        raise ValueError("x cannot be less than xmin")
    
    idx = np.int32((x-xmin)/(xmax - xmin)*nstep)
    return idx

def S(kappa, r, v, r_grid, v_grid, den, r200):
    A = calculate_A(kappa, den, r_grid, v_grid)
    
    r_min = r_grid.min()
    r_max = r_grid.max()
    r_res = r_grid.size
    
    v_min = v_grid.min()
    v_max = v_grid.max()
    v_res = v_grid.size
    
    r200_idx = coor2idx(r200, r_min, r_max, r_res)          # index of r200 in r_grid
    
    '''
    # phi is calculated by integrating f_q within the caustic lines.
    phi = np.empty(r200_idx)
    for i in range(phi.size):
        lower_limit = coor2idx(-A[i], v_min, v_max, v_res)
        upper_limit = coor2idx( A[i], v_min, v_max, v_res)
        phi[i] = np.trapz(den[lower_limit:upper_limit, i], x = v[lower_limit:upper_limit])
    #phi = np.trapz(den[:r200_idx], dx = v_grid[1] - v_grid[0], axis = -1)    # integral along v axis
    
    #v_esc_mean_squared = np.trapz((A[:r200_idx]**2) * phi, x = r_grid[:r200_idx]) / np.trapz(phi, x = r_grid[:r200_idx])
    
    
    if np.trapz(phi, dx = r_grid[1] - r_grid[0]) == 0:
        return np.inf

    v_esc_mean_squared = np.trapz((A[:r200_idx]**2) * phi, x = r_grid[:r200_idx]) / np.trapz(phi, x = r_grid[:r200_idx])
    cond_idx = (r < r200) & (abs(v) < A[coor2idx(r, r_min, r_max, r_res)])
    
    
    if not np.any(cond_idx):
        return np.inf
    
    
    v_mean_squared = np.sum((v[cond_idx])**2) / (v[cond_idx]).size
    return (v_esc_mean_squared - 4*v_mean_squared)**2
    '''
    
    # phi is calculated by integrating f_q within the caustic lines.
    # calculating v^2 from density estimation
    phi = np.empty(r200_idx)
    v_sq = np.empty(r200_idx)
    for i in range(phi.size):
        lower_limit = coor2idx(-A[i], v_min, v_max, v_res)
        upper_limit = coor2idx( A[i], v_min, v_max, v_res)
        phi[i] = np.trapz(den[lower_limit:upper_limit, i], x = v[lower_limit:upper_limit])
        v_sq[i] = np.trapz(den[lower_limit:upper_limit, i] * (v_grid[lower_limit:upper_limit])**2, x = v[lower_limit:upper_limit])
    #phi = np.trapz(den[:r200_idx], dx = v_grid[1] - v_grid[0], axis = -1)    # integral along v axis    
    
    if np.trapz(phi, x = r_grid[:r200_idx]) == 0:
        return np.inf

    v_esc_mean_squared = np.trapz((A[:r200_idx]**2) * phi, x = r_grid[:r200_idx]) / np.trapz(phi, x = r_grid[:r200_idx])
    v_mean_squared = np.trapz(v_sq, x = r_grid[:r200_idx]) / np.trapz(phi, x = r_grid[:r200_idx])
    
    return (v_esc_mean_squared - 4*v_mean_squared)**2

def S(kappa, r, v, r_grid, v_grid, den, r200):
    r_min = r_grid.min()
    r_max = r_grid.max()
    r_res = r_grid.size
    
    v_min = v_grid.min()
    v_max = v_grid.max()
    v_res = v_grid.size
    
    r200_idx = coor2idx(r200, r_min, r_max, r_res)
    
    A = calculate_A(kappa, den, r_grid, v_grid)
    
    '''
    x0Range = rRange*H
    R = (np.sqrt(5)-1)*0.5        # golden ratio

    TOL = 1e-5
    error = TOL+1
    a = np.empty(rRange.size)
    a.fill(0/q)
    b = np.empty(rRange.size)
    b.fill(vHigh/q)
    while np.all(error > TOL):
        a1 = b - R*(b - a)
        b1 = a + R*(b - a)

        a[(abs(fq(x0Range, a1, xdata_0, xdata_1, gauss2d, h)-kappa)) >  (abs(fq(x0Range, b1, xdata_0, xdata_1, gauss2d, h)-kappa))] = a1[(abs(fq(x0Range, a1, xdata_0, xdata_1, gauss2d, h)-kappa)) >  (abs(fq(x0Range, b1, xdata_0, xdata_1, gauss2d, h)-kappa))]
        b[(abs(fq(x0Range, a1, xdata_0, xdata_1, gauss2d, h)-kappa)) <= (abs(fq(x0Range, b1, xdata_0, xdata_1, gauss2d, h)-kappa))] = b1[(abs(fq(x0Range, a1, xdata_0, xdata_1, gauss2d, h)-kappa)) <= (abs(fq(x0Range, b1, xdata_0, xdata_1, gauss2d, h)-kappa))]

        error = abs(b - a)

    a[abs(fq(x0Range, a, xdata_0, xdata_1, gauss2d, h)-kappa) > TOL*kappa] = 0
    A = a*q
    '''
    
    phi = np.trapz(den[:r200_idx], x = v_grid, axis = -1)
    
    v_esc_sqaured_mean = np.trapz((A[:r200_idx]**2) * phi, dx = r_grid[1] - r_grid[0]) / np.trapz(phi, dx = r_grid[1] - r_grid[0])
    cond_idx = (r < r200) & (abs(v) < A[coor2idx(r, r_min, r_max, r_res)])
    v_squared_mean = np.sum((v[cond_idx])**2) / (v[cond_idx]).size
    return (v_esc_sqaured_mean - 4*v_squared_mean)**2
    
def calculate_A(kappa, den, r_grid, v_grid):
    '''
    # golden search for solution f(r, v) = kappa
    R = (np.sqrt(5)-1)*0.5        # golden ratio
    
    ## upper contour
    TOL = 1e-5
    error = TOL + 1
    
    a = np.empty(x_grid.size)
    a.fill(0/q)
    b = np.empty(x_grid.size)
    b.fill(v_max/q)
    while np.all(error > TOL):
        x1 = b - R*(b - a)
        x2 = a + R*(b - a)
        
        cond_idx = (abs(f(x_grid, x1) - kappa)) >  (abs(f(x_grid, x2) - kappa))

        a[cond_idx] = x1[cond_idx]
        b[np.logical_not(cond_idx)] = x2[np.logical_not(cond_idx)]

        error = abs(b - a)

    
    ### if the function values does not satisfy the TOLerance, then assume that there is no such solution, and set the value to 0
    a[abs(f(x_grid, a)-kappa) > TOL*kappa] = 0
    A_up = a*q
    
    ## lower contour
    TOL = 1e-5
    error = TOL + 1
    
    a = np.empty(x_grid.size)
    a.fill(0/q)
    b = np.empty(x_grid.size)
    b.fill(v_min/q)
    while np.all(error > TOL):
        x1 = b - R*(b - a)
        x2 = a + R*(b - a)

        cond_idx = (abs(f(x_grid, x1) - kappa)) >  (abs(f(x_grid, x2) - kappa))

        a[cond_idx] = x1[cond_idx]
        b[np.logical_not(cond_idx)] = x2[np.logical_not(cond_idx)]
        
        error = abs(b - a)
    
    
    ### if the function values does not satisfy the TOLerance, then assume that there is no such solution, and set the value to 0
    a[abs(f(x_grid, a)-kappa) > TOL*kappa] = 0
    A_down = a*q
    
    A = np.minimum(np.abs(A_up), np.abs(A_down))
    A = grad_restrict(A, r_grid)
    
    return A
    '''
    contours = skimage.measure.find_contours(den, kappa)
    r_step = r_grid[1] - r_grid[0]
    v_step = v_grid[1] - v_grid[0]
    A = np.full(r_grid.size, np.inf)
    for contour in contours:
        r_cont = contour[:,1]
        v_cont = contour[:,0]
        
        if not (r_cont[0] == r_cont[-1] and v_cont[0] == v_cont[-1]):   # consider only non-looping contours           
            '''
            #interpolate
            split_idx = (np.diff(np.sign(np.diff(r_cont))) != 0).nonzero()[0] + 2
            r_segments = np.split(r_cont, split_idx)
            v_segments = np.split(v_cont, split_idx)

            for r_seg, v_seg in zip(r_segments, v_segments):
                if (r_seg.size > 1) and (r_seg[0] > r_seg[1]):
                    r_seg = r_seg[::-1]
                    v_seg = v_seg[::-1]

                r_cont_grid = np.arange(r_seg.max()).astype(int)
                v_cont_grid = np.interp(r_cont_grid, r_seg, v_seg)
                print(r_seg)
            '''
            int_idx = (r_cont == r_cont.astype(int))
            r_cont_grid = r_cont[int_idx].astype(int)
            v_cont_grid = v_cont[int_idx]
            for r, v in zip(r_cont_grid, v_cont_grid):
                v = v*v_step + v_grid.min()
                A[r] = min(A[r], abs(v))

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
    
    '''
    if (it == 1) & (search_all):
        #search_range = np.logspace(np.log10(a + abs(b-a)*1e-6), np.log10(b), 100)
        search_range = np.linspace(a, b, 100)
        min_idx = 0
        min_val = fn(search_range[0])
        for i in range(search_range.size):
            x = search_range[i]
            val = fn(x)
            print("{} : {}".format(i, val))
            if val < min_val:
                min_idx = i
                min_val = val
        
        print("min_idx : {}, min_x : {}, min_val : {}".format(min_idx, search_range[min_idx], min_val))
        #return search_range[min_idx]
        a = max(search_range[min_idx] - (b-a)*0.25, a)
        b = min(search_range[min_idx] + (b-a)*0.25, b)
        print("search range: {} ~ {}".format(a, b))
    '''
    a_init = a
    b_init = b
    
    R = (np.sqrt(5)-1)*0.5        # golden ratio
    
    TOL = 1e-5 * abs(b - a)
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

def grad_restrict(A, r, grad_limit = 2):
    for i in range(A.size-1):
        if (A[i] <= 0) or (A[i+1] <= 0) or (r[i] == 0):
            continue
        
        #if A[i+1] <= 0:
        #    A[i+1] = np.exp( np.log(A[i]) - grad_limit*(np.log(r[i+1]) - np.log(r[i])) )
        
        else:
            log_grad = (np.log(A[i+1])-np.log(A[i])) / (np.log(r[i+1]) - np.log(r[i]))
            if log_grad > grad_limit:
                A[i+1] = np.exp( np.log(A[i]) + 0.5*(np.log(r[i+1]) - np.log(r[i])) )

            #elif log_grad < -grad_limit:
            #    A[i+1] = np.exp( np.log(A[i]) - grad_limit*(np.log(r[i+1]) - np.log(r[i])) )
            
            
    return A

