import numpy as np

def density_estimation(x_data, y_data, x_res, y_res, alpha, h_c, display_log=False):

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

        # For adaptive kernel density estimation, we need to calculate various bandwidth factors.
        N = x_data.size                                                                                                 # number of data points
        h_opt = 6.24/(N**(1/6)) * np.sqrt((np.std(x_data)**2 + np.std(y_data)**2)/2)                                    # eq. 20 from Serra et al. 2011
        
        gamma = 10**(np.sum(np.log10(fq(x_data, y_data, x_data, y_data, triweight, h_opt)))/N)                # gamma  defined in Diaferio 1999, between eq. 17 and eq. 18; Here, the term is divided by 2*N because N is the number of original (i.e. un-mirrored) data points
        lam = np.sqrt(gamma/fq(x_data, y_data, x_data, y_data, triweight, h_opt))                             # lambda defined in Diaferio 1999, between eq. 17 and eq. 18

        fn = lambda h_c: M_0(h_c, h_opt, lam, x_data, y_data)                                                      # M_0 function defined in eq. 18 from Diaferio 1999; Here we used the lambda function to fix input arguments other than h_c.
        if h_c is None:
            if display_log == True:
                print("Calculating h_c.")
            h_c = find_hc(h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data, x_res=x_res, y_res=y_res, display_log=display_log)                                 # find h_c that minimizes M_0
            if display_log == True:
                print("h_c = {:.5e}".format(h_c))
        else:
            if display_log == True:
                print("User-given h_c = {:.5e}".format(h_c))
        h_c = h_c*alpha
        if display_log == True:
            print("final value of h_c = {:.5e}".format(h_c))

        
        h = h_c * h_opt * lam                                                                                      # final h_i (local smoothing length); size of h_i is same as x_data and y_data

        return lambda x, y: fq(x, y, x_data, y_data, triweight, h)

def triweight(x, y):

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

def fq(x, y, x_data, y_data, K, h):

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

def find_hc(h_opt, lam, x_data, y_data, x_res, y_res, display_log):
        
    '''
    Finds optimal h_c value for kernel bandwidth used in kernel density estimation.
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
    M0_0 = M_0(h_c=hc_0, h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data, x_res=x_res, y_res=y_res)
    M0_1 = M_0(h_c=hc_1, h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data, x_res=x_res, y_res=y_res)
    if(display_log):
        print("Iteration   1, hc = {:.7f}: M_0 = {:.7e}".format(hc_0, M0_0))
        print("Iteration   2, hc = {:.7f}: M_0 = {:.7e}".format(hc_1, M0_1))
    
    i = 2
    while(i < max_guess):
        hc_2 = hc_1 + step_size
        M0_2 = M_0(h_c=hc_2, h_opt=h_opt, lam=lam, x_data=x_data, y_data=y_data, x_res=x_res, y_res=y_res)
        
        if(display_log):
            print("Iteration {:3}, hc = {:.7f}: M_0 = {:.7e}".format(i+1, hc_2, M0_2))
        
        if (M0_2 > M0_1):
            # Find minimum by parabolic 
            h_c = parabola_min(hc_0, hc_1, hc_2, M0_0, M0_1, M0_2)
            
            if(display_log):
                print("Optimal h_c: {}".format(h_c))
            
            return h_c
            
        else:
            i += 1
            hc_0 = hc_1
            hc_1 = hc_2
            M0_0 = M0_1
            M0_1 = M0_2
    
    raise Exception("Failed to find optimal kernel size hc.")

def M_0(h_c, h_opt, lam, x_data, y_data, x_res, y_res):

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
    #### set up grids for numerical integration; x_grid and y_grid here is different from those used in the main function.
    #### Note that the value of triweight funtion is 0 outside (x/h)**1 + (y/h)**1 = 1
    x_min = 0                                      # minimum value in the x_grid
    x_max = np.max(x_data + h)                     # maximum value in the x_grid
    y_min = np.min(y_data - h)                     # minimum value in the y_grid
    y_max = np.max(y_data + h)                     # maximum value in the y_grid

    x_grid = np.linspace(x_min, x_max, x_res)      # grid along rescaled r-axis (x-axis) used for numerical integration
    y_grid = np.linspace(y_min, y_max, y_res)      # grid along rescaled v-axis (y-axis) used for numerical integration

    X, Y = np.meshgrid(x_grid, y_grid)             # mesh grid
    
    f_squared = fq(X, Y, x_data, y_data, triweight, h)**2                   # squared value of fq calcuated at each point (X, Y)

    term_1 = np.trapz(np.trapz(f_squared, x=y_grid, axis=0), x=x_grid)      # first term of M_0 is the integration of fq squared
    
    # calculating the second term (refer to eq. 3.37 and Section 5.3.4 of Silverman B. W., 1986, Density Estimation for Statistics and Data Analysis, Chapman & Hall, London)
    term_2 = 0
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        term_2 += fq(x=x_data[i], y=y_data[i], x_data=x_data[mask], y_data=y_data[mask], K=triweight, h=h[mask])
    term_2 *= 2/N
    
    return term_1 - term_2                                              # return value: M_0 function evaluated for given h_c

def parabola_min(x0, x1, x2, y0, y1, y2):
        
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