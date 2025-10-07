# This function constructs a binary tree from galaxies using hierarchical clustering,
# and then determines candidate members of a galaxy cluser.
# Although the candidate members do not perfectly match with the actual members,
# they are useful in cases such as when determining the size of the cluster.
# See Section 4.1 of Serra et al. 2011, MNRAS.

import numpy as np
import scipy.cluster
def hier_clustering(gal_ra, gal_dec, gal_v, threshold="ALS", gal_m=1e12, mask=None, sig_pl=None):
    '''
    Constructs binary tree using hierarchical clustering and finds candidate member galaxies.
    This function is based on Diaferio 1999 and Serra et al. 2011.

    Parameters
    ---------------------------
    gal_ra      : numpy ndarray,    RA of galaxies (in units of deg)
    gal_dec     : numpy ndarray,    Dec of galaxies (in units of deg)
    gal_v       : numpy ndarray,    l.o.s. vel of galaxies (in units of km/s)
    threshold   : str,              threshold of cutting the binary tree; "AD" for Antonaldo Diaferio (explicated in Appendix A, Diaferio 1999)
                                    and "ALS" for Ana Laura Serra (explicated in Section 4.2, Serra et al. 2011);
                                    for the moment, only ALS method is available; default value "ALS"
    gal_m       : float,            mass of a single galaxy (in units of solar mass); default value 1e12
    mask        : numpy ndarray,    mask applied to gal_ra, gal_dec, and gal_v; if None, masking is not applied; default value None

    Returns
    ---------------------------
    list of 0s and 1s, where 0 is for non-candidate members and masked members and 1 is for candidate members
    '''

    N_total = len(gal_ra)                   # total number of galaxies before making is applied
    # Apply masking, if there is any.
    if mask is not None:
        gal_ra  = gal_ra[mask]
        gal_dec = gal_dec[mask]
        gal_v   = gal_v[mask]

    N = len(gal_ra)                         # number of galaxies

    # Constants.
    m = gal_m  * 2e30                       # mass of a single galaxy (in units of kg) (1 solar mass = 2e30 kg)
    G = 6.67e-11                            # gravitational constant  (in units of N m^2 kg^-2)
    c = 299792.458                          # speed of light          (in units of km/s)
    H0 = 100                                # Hubble constant         (in units of km/s/Mpc)
    Mpc = 3.086e22                          # conversion factor from Mpc to meter
    km = 1000                               # conversion factor from km to meter

    # Calculate pairwise binding energy.
    zi = gal_v/c                                    # redshift of each galaxy
    r_m = 2*c/H0 * (1 - 1/np.sqrt(1 + zi)) * Mpc    # distance corresponding to redshift zi [m] (eq. 10 from Serra et al. 2011)

    gal_ra_rad = gal_ra * np.pi/180                 # RA of each galaxies in radians
    gal_dec_rad = gal_dec * np.pi/180               # Declination of each galaxies in radians

    ri = np.stack([  r_m*np.cos(gal_dec_rad)*np.cos(gal_ra_rad),   r_m*np.cos(gal_dec_rad)*np.sin(gal_ra_rad),   r_m*np.sin(gal_dec_rad)], axis=-1)             # vector r_i from Serra et al. 2011 (in units of m)
    vi = np.stack([gal_v*np.cos(gal_dec_rad)*np.cos(gal_ra_rad), gal_v*np.cos(gal_dec_rad)*np.sin(gal_ra_rad), gal_v*np.sin(gal_dec_rad)], axis=-1) * km        # velocity vector in the same direction as ri and with magnitude gal_v (in units of m/s)

    rij = np.stack([np.subtract.outer(ri[:,0], ri[:,0]), np.subtract.outer(ri[:,1], ri[:,1]), np.subtract.outer(ri[:,2], ri[:,2])], axis=-1)                    # vector r_12 from Serra et al. 2011 (in units of m)
    vij = np.stack([np.subtract.outer(vi[:,0], vi[:,0]), np.subtract.outer(vi[:,1], vi[:,1]), np.subtract.outer(vi[:,2], vi[:,2])], axis=-1)                    # vector v_i - v_j (in units of m/s)
    lij = np.stack([     np.add.outer(ri[:,0], ri[:,0]),      np.add.outer(ri[:,1], ri[:,1]),      np.add.outer(ri[:,2], ri[:,2])], axis=-1)/2                  # vector l_12 from Serra et al. 2011 (in units of m)

    pi = (rij[:,:,0]*lij[:,:,0] + rij[:,:,1]*lij[:,:,1] + rij[:,:,2]*lij[:,:,2]) / np.sqrt(lij[:,:,0]**2 + lij[:,:,1]**2 + lij[:,:,2]**2)                     # pi from eq. 12 of Serra et al. 2011 in dimension of distance (in units of m)
    rp = np.sqrt(rij[:,:,0]**2 + rij[:,:,1]**2 + rij[:,:,2]**2 - pi**2)                                                                                       # r_p from eq. 13 of Serra et al. 2011 (in units of m)

    zlplus1 = (0.5* np.add.outer(1/np.sqrt(1+zi), 1/np.sqrt(1+zi)))**(-2)                                                                                       # z_l + 1, with z_l satisfying r_l = (r_1 + r_2)/2
    Rp = rp/zlplus1                                                                                                                                             # R_p from Serra et al. 2011 (in units of m)
    Pi = pi/zlplus1 * H0*km/Mpc                                                                                                                               # Pi  from Serra et al. 2011 (in units of m/s)

    di = np.diag_indices(N)                             # diagonal indices of N by N square matrix
    Rp[di] = 1                                          # arbitrarily set diagonal of Rp as 1, because its value is 0.
    Pi[di] = 1                                          # arbitrarily set diagonal of Pi as 1, because its value is 0.

    E = -G * m*m/Rp + 0.5 * m*m/(m+m) *Pi**2            # binding energy (i.e. measure of similarity) (in units of J)
    E[di] = 0                                           # set diagoal of E as 0; this is necessary for scipy.clustering to work.


    # Group galaxies.
    # To use scipy.cluster.hierarchy, distance matrix must be given in form of a dense matrix.
    E_dense = scipy.spatial.distance.squareform(E)                          # squareform function converts redundante matrix into dense matrix
    Z = scipy.cluster.hierarchy.single(E_dense)
    # The 1st and 2nd columns of Z[i] are the two groups that are combined at iteration i.
    # The 3rd column is the distance (or here, the binding energy) betweeen the two groups being merged.
    # The 4th column is the number of leaf nodes hanging from the newly created node.
    # We do not need the 3rd column.
    # And because it contains float values, other columns, which store node indices.
    # This causes indexing problems later on. 
    # So it is best to remove the 3rd column and change the dtype to int.
    Z = Z[:, [0, 1, 3]].astype(int)

    nnode = Z.shape[0]                  # number of nodes (except for the leaf nodes).
                                        # Leaf nodes range from 0 to N-1.
                                        # Interim nodes range from N to nnode+N-2.
                                        # Root node is nnode+N-1.
    count_leaf = Z[:,2]                 # Number of leaf nodes hanging from each node.
    leaves = [[]]*nnode                 # 2D list storing the leaf nodes hanging each node.
    for i in range(nnode):
        leaf = []                       # 1D list storing the leaf nodes hanging from node i+N; this list will be assigned to leaves[i].
        lchild, rchild = Z[i,0:2]       # left and right children of node i+N.
        if lchild < N:                  # Case: left child is a leaf node.
            leaf += [lchild]            ##### Left child is added to the list of leaf nodes hanging from node i+N.
        else:                           # Case: left child is not a leaf node.
            leaf += leaves[lchild-N]    ##### Leaf nodes hanging from left child is added to the list of leaf nodes haning from node i+N.

        if rchild < N:                  # Case: right child is a leaf node.
            leaf += [rchild]            ##### Righ child is added to the list of leaf nodes hanging from node i+N.
        else:                           # Case: right child is not a leaf node.
            leaf += leaves[rchild-N]    ##### Leaf nodes hanging from right child is added to the list of nodes hanging from node i+N.
        
        leaves[i] = leaf

    # determin the main branch.
    mainbranch = np.asarray([nnode+N-1])                            # array storing the nodes on the mainbranch; The root node is initially stored in the array.

    i = nnode+N-1                                                   # initial node of the branch (root node)
    while(True):                                                    # The while loop iterates only if iterate is True, i.e. the left and right children of the root node are not leaf nodes.
        lchild, rchild = Z[i-N, 0:2]                                # left and right children of the node i.
        if lchild < N:                                              # Case: left child is a leaf node.
            i = rchild                                              ##### The next node is set to the right child.
            mainbranch = np.append(mainbranch, [rchild])            ##### Right child is added to the mainbranch, i.e. the mainbranch follows the right child.
            if rchild < N:                                          ##### Case: right child is a leaf node.
                break                                               ########## Reached the bottom of the binary tree.

        elif count_leaf[lchild - N] > count_leaf[rchild - N]:       # Case: # of leaf nodes hanging from the left node is greater than # of leaf nodes hanging from the right node.
            i = lchild                                              ##### The next node is set to the left child.
            mainbranch = np.append(mainbranch, [lchild])            ##### Left child is added to the mainbranch, i.e. the mainbranch follows the left child.
        else:                                                       # Case: # of leaf nodes hanging from the right node is greater than or equal to # of leaf nodes hanging from the left node.         
            i = rchild                                              ##### The next node is set to the right child.
            mainbranch = np.append(mainbranch, [rchild])            ##### Right child is added to the mainbranch, i.e. the mainbranch follows the right child.

    mainbranch = mainbranch[:-1]                                    # exclude the last node, which is a leaf node.

    sigma = np.zeros(mainbranch.size)                               # velocity dispersion at each level of the main branch.
    for i, node in enumerate(mainbranch):
        sigma[i] = np.std(gal_v[leaves[node-N]])


    hist, bins = np.histogram(sigma, bins=np.arange(np.min(sigma), np.max(sigma), (np.max(sigma)-np.min(sigma))/5))                   # apply bins to get the mode
    if sig_pl is None:
        sig_pl = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1])/2    # mode of the velocity dispersions

    # find where to cut the binary tree
    if threshold == "AD":
        cut_idx = ADcut(mainbranch, count_leaf, N, Z, sigma, sig_pl)
    elif threshold == "ALS":
        cut_idx = ALScut(sigma, sig_pl)

    cand_mem_idx = leaves[mainbranch[cut_idx]-N]                    # indices of candidate members are the leaves hanging from the cut

    cand_mem = np.zeros(N).astype(bool)
    cand_mem[cand_mem_idx] = True                                      # assign value 1 to candidate members

    total_cand_mem = np.zeros(N_total).astype(bool)
    total_cand_mem[mask] = cand_mem
    
    return total_cand_mem, mainbranch, sigma, cut_idx

def ADcut(mainbranch, count_leaf, N, Z, sigma, sig_pl, f=0.1):
    thresholds = np.asarray([0])
    for i in range(mainbranch.size):
        node = mainbranch[i]
        lchild, rchild = Z[node-N, 0:2]
        if lchild < N or rchild < N:
            continue
        
        nd = count_leaf[node-N]
        ns = min(count_leaf[lchild-N], count_leaf[rchild-N])
        if ns > f*nd:
            thresholds = np.append(thresholds, i)
    
    if np.where(np.abs(sig_pl - sigma[thresholds])/sig_pl < 0.1)[0].size == 0:
        cut_idx = 0
        print("No cut!")
    else:
        cut_idx = np.where(np.abs(sig_pl - sigma[thresholds])/sig_pl < 0.3)[0][0]
    
    return cut_idx
        
            

def ALScut(sigma, sig_pl):
    N03 = np.sum(np.abs(sig_pl - sigma)/sig_pl < 0.3)                               # N_0.3 described in Section 4.2 of Serra et al. 2011.

    delta = 0.03
    while (delta < 0.3):
        N_del = np.sum(np.abs(sig_pl - sigma)/sig_pl < delta)                       # number of nodes with in delta range
        if N_del >= 0.8*N03:
            break
        
        elif delta < 0.1:                                                           # delta ranges from 0.03 to 0.1
            delta += 0.01                                                           # in step size of 0.01
        
        else:                                                                       # however, if N_del does not reach N03 for delta in [0.03, 0.1]
            delta += 0.001                                                          # we need to find when N_del reaches N03

    
    if np.sum(np.abs(sig_pl - sigma)/sig_pl < delta) < 5:                           # If we cannot find a plateau of large enouge size,
        cut_idx = 0                                                                 # then we treat the first node as the cut.
    else:
        cand_cut_idx = np.where(np.abs(sig_pl - sigma)/sig_pl < delta)[0][0:5]      # candidate of cuts are the first five nodes within the delta range
        cut_idx = cand_cut_idx[np.argmin(np.abs(sig_pl - sigma[cand_cut_idx]))]     # the final cut is selected as the one with the least discrepancy from the sigma plateau

    return cut_idx