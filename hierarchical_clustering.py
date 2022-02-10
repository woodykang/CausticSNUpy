import numpy as np
import scipy.cluster

# This function constructs a binary tree from galaxies using hierarchical clustering,
# and then determines candidate members of a galaxy cluser.
# Although the candidate members do not perfectly match with the actual members,
# they are useful in cases such as when determining the size of the cluster.
# See Section 4.1 of Serra et al. 2011, MNRAS.


def hier_clustering(gal_ra, gal_dec, gal_v, threshold="AD"):

    ################### INPUTS ###################
    # gal_ra  : RA of galaxies in degree
    # gal_dec : Dec of galaxies in degree
    # gal_v   : l.o.s. velocities of galaxies in km/s
    ################### RETURN ###################
    # member_idx : indices of candidate members
    ################# REFERENCE ##################
    # Serra et al. 2011, MNRAS
    ##############################################

    N = len(gal_ra)                         # number of galaxies

    # constants
    m = 1e12  * 2e30                        # mass of a single galaxy [kg] (1 solar mass = 2e30 kg)
    G = 6.67e-11                            # gravitational constant  [N m^2 kg^-2]
    c = 299792458                           # speed of light          [km/s]
    H0 = 100                                 # Hubble constant         [km/s/Mpc]
    Mpc = 3.086e22                          # conversion factor from Mpc to meter
    km = 1000                               # conversion factor from km to meter

    # calculate pairwise binding energy
    zi = gal_v/c                                    # redshift of each galaxy
    r_m = 2*c/H0 * (1 - 1/np.sqrt(1 + zi)) * Mpc     # distance corresponding to redshift zi [m] (eq. 10 from Serra et al. 2011)

    gal_ra_rad = gal_ra * np.pi/180                 # RA of each galaxies in radians
    gal_dec_rad = gal_dec * np.pi/180               # Declination of each galaxies in radians

    ri = np.stack([  r_m*np.cos(gal_dec_rad)*np.cos(gal_ra_rad),   r_m*np.cos(gal_dec_rad)*np.sin(gal_ra_rad),   r_m*np.sin(gal_dec_rad)], axis=-1)             # vector r_i from Serra et al. 2011 [m]
    vi = np.stack([gal_v*np.cos(gal_dec_rad)*np.cos(gal_ra_rad), gal_v*np.cos(gal_dec_rad)*np.sin(gal_ra_rad), gal_v*np.sin(gal_dec_rad)], axis=-1) * km        # velocity vector in the same direction as ri and with magnitude gal_v [m/s]

    rij = np.stack([np.subtract.outer(ri[:,0], ri[:,0]), np.subtract.outer(ri[:,1], ri[:,1]), np.subtract.outer(ri[:,2], ri[:,2])], axis=-1)                    # vector r_12 from Serra et al. 2011 [m]
    vij = np.stack([np.subtract.outer(vi[:,0], vi[:,0]), np.subtract.outer(vi[:,1], vi[:,1]), np.subtract.outer(vi[:,2], vi[:,2])], axis=-1)                    # vector v_i - v_j [m/s]
    lij = np.stack([     np.add.outer(ri[:,0], ri[:,0]),      np.add.outer(ri[:,1], ri[:,1]),      np.add.outer(ri[:,2], ri[:,2])], axis=-1)/2                  # vector l_12 from Serra et al. 2011 [m]

    pi_r = (rij[:,:,0]*lij[:,:,0] + rij[:,:,1]*lij[:,:,1] + rij[:,:,2]*lij[:,:,2]) / np.sqrt(lij[:,:,0]**2 + lij[:,:,1]**2 + lij[:,:,2]**2)                     # pi from eq. 12 of Serra et al. 2011 in dimension of distance [m]
    pi_v = (vij[:,:,0]*lij[:,:,0] + vij[:,:,1]*lij[:,:,1] + vij[:,:,2]*lij[:,:,2]) / np.sqrt(lij[:,:,0]**2 + lij[:,:,1]**2 + lij[:,:,2]**2)                     # pi from eq. 12 of Serra et al. 2011 in dimension of velocity [m/s]

    rp = np.sqrt(rij[:,:,0]**2 + rij[:,:,1]**2 + rij[:,:,2]**2 - pi_r**2)                                                                                       # r_p from eq. 13 of Serra et al. 2011 [m]

    zlplus1 = (0.5* np.add.outer(1/np.sqrt(1+zi), 1/np.sqrt(1+zi)))**(-2)                                                                                       # z_l + 1, with z_l satisfying r_l = (r_1 + r_2)/2
    Rp = rp/zlplus1                                                                                                                                             # R_p from Serra et al. 2011 [m]
    Pi = pi_v/zlplus1                                                                                                                                           # Pi  from Serra et al. 2011 [m/s]

    di = np.diag_indices(N)                             # diagonal indices of N by N square matrix
    Rp[di] = 1                                          # arbitrarily set diagonal of Rp as 1, because its value is 0.
    Pi[di] = 1                                          # arbitrarily set diagonal of Pi as 1, because its value is 0.

    E = -G * m*m/Rp + 0.5 * m*m/(m+m) *Pi**2            # binding energy (i.e. measure of similarity)
    E[di] = 0                                           # set diagoal of E as 0; this is necessary for scipy.clustering to work.


    # group galaxies
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
            leaf += leaves[lchild-N]    #### Leaf nodes hanging from left child is added to the list of leaf nodes haning from node i+N.

        if rchild < N:                  # Case: right child is a leaf node.
            leaf += [rchild]            #### Righ child is added to the list of leaf nodes hanging from node i+N.
        else:                           # Case: right child is not a leaf node.
            leaf += leaves[rchild-N]    #### Leaf nodes hanging from right child is added to the list of nodes hanging from node i+N.
        
        leaves[i] = leaf

    # determin the main branch.
    mainbranch = np.asarray([nnode+N-1])                            # array storing the nodes on the mainbranch; The root node is initially stored in the array.

    i = nnode+N-1                                                   # initial node of the branch (root node)
    while(True):                                                    # The while loop iterates only if iterate is True, i.e. the left and right children of the root node are not leaf nodes.
        lchild, rchild = Z[i-N, 0:2]                                  # left and right children of the node i.
        if lchild < N:                                              # Case: left child is a leaf node.
            i = rchild                                              #### The next node is set to the right child.
            mainbranch = np.append(mainbranch, [rchild])            #### Right child is added to the mainbranch, i.e. the mainbranch follows the right child.
            if rchild < N:                                          #### Case: right child is a leaf node.
                break                                               ######### Reached the bottom of the binary tree.

        elif count_leaf[lchild - N] > count_leaf[rchild - N]:       # Case: # of leaf nodes hanging from the left node is greater than # of leaf nodes hanging from the right node.
            i = lchild                                              #### The next node is set to the left child.
            mainbranch = np.append(mainbranch, [lchild])            #### Left child is added to the mainbranch, i.e. the mainbranch follows the left child.
        else:                                                       # Case: # of leaf nodes hanging from the right node is greater than or equal to # of leaf nodes hanging from the left node.         
            i = rchild                                              #### The next node is set to the right child.
            mainbranch = np.append(mainbranch, [rchild])            #### Right child is added to the mainbranch, i.e. the mainbranch follows the right child.

    mainbranch = mainbranch[:-1]                    # exclude the last node, which is a leaf node.

    sigma = np.zeros(mainbranch.size)               # velocity dispersion at each level of the main branch.
    for i, node in enumerate(mainbranch):
        sigma[i] = np.std(gal_v[leaves[node-N]])


    hist, bins = np.histogram(sigma, bins='auto')                   # apply bins to get the mode
    sig_pl = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1])/2    # mode of the velocity dispersions

    # find where to cut the binary tree
    if threshold == "AD":
        cut_idx = ADcut(mainbranch, count_leaf, N, Z, sigma, sig_pl)
    elif threshold == "ALS":
        cut_idx = ALScut(sigma, sig_pl)
    
    cand_mem_idx = leaves[mainbranch[cut_idx]-N]

    cand_mem = np.zeros(N)
    cand_mem[cand_mem_idx] = 1
    
    return cand_mem_idx

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
    N03 = (np.abs(sig_pl - sigma)/sig_pl < 0.3).size                # N_0.3 described in Section 4.2 of Serra et al. 2011.

    delta = 0.03                                                    # delta ranges from 0.03 to 0.1
    while (delta < 0.1):
        N_del = (np.abs(sig_pl - sigma)/sig_pl < delta).size
        if N_del >= 0.8*N03:
            break
        
        else:
            delta += 0.01

    if delta == 0.1:                # no plateau is detected
        cut_idx = 0

    else:
        if np.where(np.abs(sig_pl - sigma)/sig_pl < delta)[0].size < 10:
            cut_idx = 0
        else:
            cut_idx = np.where(np.abs(sig_pl - sigma)/sig_pl < delta)[0][0]
        
    return cut_idx