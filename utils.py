import networkx as nx
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import warnings
from matplotlib import pyplot as plt
import numpy.linalg as linalg
from scipy.linalg import norm as norm_sp
import time
import copy
import scipy.stats as stat
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import SpectralClustering


######SLEM CODE##############

'''Code adapted from yang to accomodate a weighted adjacency matrix
 @mastersthesis{yang2015numerical,
  title={Numerical Methods for Solving the Fastest Mixing Markov Chain Problem},
  author={Yang, Yang Kjeldsen},
  year={2015}
}
'''

def tp_rw_matrix_weighted ( graph, W) :
    '''Compute standard random walk matrix 
    with entry P[i,j] denoting probability of moving from 
    vertex i to j'''
    edges = list(graph . edges ())
    n = graph . number_of_nodes ()
    P = np . zeros (( n , n ) )
    for i , j in edges :
        d_i = W[i].sum()
        d_j = W[j].sum()
        P [i , j ] = W[i,j]/float(d_i)
        P [j , i ] = W[j,i]/float(d_j)
    return P

def mh_chain_weighted(graph, W):
    '''Compute metropolis hastings chain using a possibly weighted adjacency
    matrix W'''
    n = graph . number_of_nodes ()
    #unif distribution
    pi_vec = 1./ n * np . ones ( n )
    edges = graph . edges ()
    R = np . zeros (( n , n ) )
    P = np . zeros (( n , n ) )
    P_rw = tp_rw_matrix_weighted( graph, W )
    #make symmetric
    for i , j in edges :
        P [i , j ] = min(P_rw[i,j],P_rw[j,i])
        P [j, i] = P[i,j]
    #overflow on self loops to make this a distribution
    for i in range ( n ) :
        s = 0
        for k in graph . neighbors ( i ) :
            s += max(0,P_rw[i,k]-P_rw[k,i])
        P [i , i ] = P_rw [i , i ] + s
    return P

def slem (P) :
    eig_vals , eig_vecs = linalg.eig(P)
    eig_vals = list(eig_vals)
    eig_vals.sort()
    return max( -eig_vals [0], eig_vals [ -2]).real


def tp_matrix ( graph , p ) :
    '''Construct transition probability matrix from edge transition probabilities in p'''
    '''Starts with all self loops, and then as we find transitions outside of 
    each vertex removes that from the self loop'''
    edges = list(graph . edges ())
    n = graph . number_of_nodes ()
    P = np . identity ( n )
    for l in range (len ( edges ) ) :
        i , j = edges [ l ]
        P[i,j]=P[i,j]+p[l]
        P[j,i]=P[j,i]+p[l]
        P[i,i]=P[i,i]-p[l]
        P[j,j]=P[j,j]-p[l]
    return P

def sub_from_tpMatrix(graph,P):
    '''Compute sub gradient for how much slem will change with entry of P '''
    edges = list(graph . edges ())
    m = graph . number_of_edges ()
    g = np . zeros ( m )
    eig_vals , eig_vecs = linalg. eig ( P )
    eig_list = zip( eig_vals , np . transpose ( eig_vecs ) )
    eig_list  = sorted (eig_list, key = lambda x : x [0])
    lambda_2 , lambda_n = eig_list [ -2][0] , eig_list [0][0]
    if lambda_2 >= - lambda_n :
        u = [ u_i . real for u_i in eig_list [ -2][1]]
        for l in range ( m ) :
            i , j = edges [ l ]
            g [ l ] = -( u [ i ] - u [ j ]) **2
    else :
        v = [ v_i . real for v_i in eig_list [0][1]]
        for l in range ( m ) :
            i , j = edges [ l ]
            g [ l ] = ( v [ i ] - v [ j ]) **2
    return g

def solve (graph , p0 , w, max_iter =100 , alpha = lambda k : 1./( np . sqrt ( k ) ) ):
    '''p0 is the beginning transition probabilities'''
    '''w are the edge weights'''
    p = p0
    edges = list(graph . edges ())
    n = graph . number_of_nodes ()
    m = graph . number_of_edges ()
    k = 1
    P = tp_matrix(graph , p )
    sol = {'f': slem ( P ) ,
        'p': copy.copy ( p ) , 'iter' : 0 ,
        'fk': np . zeros ( max_iter +1) }
    #Computes slem
    sol ['fk'][0] = slem ( P )
    while k <= max_iter :
        # subgradient step
        g = sub_from_tpMatrix(graph,P)
        # sequential projection step
        p -= alpha ( k ) / linalg. norm ( g ) * g
        #do not assign any negative transitions
        for l in range ( m ) : p [ l ] = max ( p [ l ] , 0)
        #p can be at most w
        for l in range (m): p[l] = min(p[l],w[l])
        #Make each nodes distribution add up to at most 1
        for i in range ( n ) :
            #get indices of edges incident to i
            I = [ l for l in range ( m ) if i in edges[ l ]]
            #Substract from positive transition proabbilities 
            #until sum is at most 1
            while sum ([ p [ l ] for l in I ]) > 1:
                #Get indices of edges incident to i 
                #with positive transitions probabilities
                I = [ l for l in I if p [ l ] > 0]
                p_min = min ([ p [ l ] for l in I ])
                p_sum = sum ([ p [ l ] for l in I ])
                #Compute maximum we can subtract from all values in I
                #without any going negative
                delta = min( p_min , ( p_sum - 1.)/len( I ) )
                for l in I : p [ l ] -= delta
        #Update P
        P = tp_matrix(graph , p )
        #Record slem of kth iteration
        sol ['fk'][ k ] = slem(P)
        #If kth iteration is better, update solution
        if sol['fk'][k] < sol['f']:
            sol ['f'] = sol['fk'][k]
            sol ['p'] = copy.copy( p )
            sol ['iter'] = k
        k += 1
    return sol

def graph_values_to_vector ( graph , P ) :
    edges = list(graph . edges ())
    m = graph . number_of_edges ()
    p = np . zeros ( m )
    for l in range ( m ) :
        (i,j) = edges[l]
        p [ l ] = P [i , j ]
    return p

def optimize ( graph , W, chain = mh_chain_weighted, max_iter =200 , alpha = lambda k : 1./ np . sqrt ( k ) ):
    '''graph is a network graph'''
    '''W is adjacency numpy matrix, possibly weighted'''
    #Initialize with metropolis-hastings chain
    P = chain ( graph, W )
    #Optimization is on entries of P corresponding to edges in graph, 
    #transform entries of P and W to vectors using P and W 
    p = graph_values_to_vector ( graph , P )
    w = graph_values_to_vector (graph, W)
    sol = solve ( graph ,p , w, max_iter , alpha )
    return sol


def fast_mix_transition_matrix_slem(A):
    '''Compute FMMC matrix of A'''
    G = nx.from_numpy_matrix(A)
    sol = optimize (G , A, max_iter =60)
    #Optimum slem found
    mu = sol ['f']
    #edge probabilities at optimum
    p = sol ['p']
    P = tp_matrix (G , p )
    return P, mu

###############################
#GRAPH UTILS#

def shortestpath(A,G=None, cutoff = 100):
    '''Return sorted list of shortest path lengths up to 
    length cutoff from graph defined
    by adjacency matrix A.

    Args:

    A -- adjacency matrix 
    G -- graph defined by adjacency matrix (default is None)
    cutoff -- longest path length to compute (default is 100) 

    '''
    if G is None:
        G = nx.from_numpy_matrix(A).to_undirected()
    #Returns map of nodes to their shortest path lengths to all nodes in the graphs
    length_dict = dict(nx.all_pairs_shortest_path_length(G,cutoff=cutoff))
    shortestPathLengths = []
    vertices = length_dict.keys()
    for j in vertices:
        #Add shortest path lengths to list for each node
        shortestPathLengths = shortestPathLengths + list(length_dict[j].values())
    return sorted(shortestPathLengths)

def cc(A,G=None):
    '''Return sorted list of clustering coefficients for each node in G
    defined by A

    Args:

    A -- adjacency matrix 
    G -- graph defined by adjacency matrix (default is None)

    '''
    if G is None:
        G = nx.from_numpy_matrix(A).to_undirected()
    clustering_coefficient = nx.clustering(G)
    clustering_coefficient = sorted(list(clustering_coefficient.values()))
    return sorted(clustering_coefficient)

def assort(A,G=None):
    '''Compute assortativity of graph G defined by A. If A is empty, return 1 
    Args:

    A -- adjacency matrix 
    G -- graph defined by adjacency matrix (default is None)

    '''
    if A.sum() == 0:
        return 1
    if G is None:
        G = nx.from_numpy_matrix(A).to_undirected()
    return nx.degree_assortativity_coefficient(G)

def betweenness(A,G=None,approx_factor = 100):
    '''Approximate betweenness for each node using approx_factor
    number of nodes as pivots, see networkx for more details. 

    Args:

    A -- adjacency matrix 
    G -- graph defined by adjacency matrix (default is None)
    approx_factor -- longest path length to compute (default is 100) 

    '''
    if G is None:
        G = nx.from_numpy_matrix(A).to_undirected()
    num_k = min(approx_factor,A.shape[0])
    b_dict = dict(nx.betweenness_centrality(G,num_k))
    return sorted(b_dict.values())

def connected_component_subgraphs(G):
    '''Returns generator for connected components in graph G'''
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def lcc(A,G=None):
    '''Return largest connected component in graph 
    G defined by A
    Args:

    A -- adjacency matrix 
    G -- graph defined by adjacency matrix (default is None)

    '''
    if G is None:
        G = nx.from_numpy_matrix(A).to_undirected()
    LCC = max(connected_component_subgraphs(G), key=len)
    return LCC

def size_lcc(A,G=None):
    '''Number of nodes in largest connected component in graph
    G defined by A

    Args:

    A -- adjacency matrix 
    G -- graph defined by adjacency matrix (default is None)

    '''
    LCC = lcc(A,G)
    return len(LCC.nodes())

def centralityVector(A):
    '''Return eigenvector corresponding to the largest eigenvalue in A

    Args:

    A -- adjacency matrix 


    '''
    return kthEigenVector(A,A.shape[0]-1)

def degree_sequence(A):
    '''Compute degree sequene of G defined by A. 
    Args:

    A -- adjacency matrix 

    '''
    G = nx.from_numpy_matrix(A).to_undirected()
    degree_sequence = sorted(dict(G.degree()).values())
    return degree_sequence


######DISTRIBUTION AND VECTOR METRICS######


def gen_hist(values,slist):
    '''Return list of fraction of entries in slist that take 
    each value in values

    Args:

    values -- list of floats
    slist -- list of floats 

    '''
    slist = sorted(slist)
    values = sorted(values)
    numValues = len(values)
    lenList = float(len(slist))
    result = [0]*numValues
    for v in range(numValues):
        cnt = slist.count(values[v])
        result[v]=cnt/lenList
    return result

def emdLinear(hist1,hist2,values):
    '''Given two histograms on a list of sorted values,
    compute the earth mover's distance between the two histograms.

    Args:

    hist1, hist2 -- histograms
    values - sortest list of values that the histograms are on.

    '''
    assert(values == sorted(values))
    numBins = len(hist1)
    assert(len(hist2) == numBins)
    earth = 0
    totalDist = 0
    for i in range(numBins-1):
        earth = (hist1[i]+earth)-hist2[i]
        dist = values[i+1]-values[i]
        totalDist = totalDist+(abs(earth)*dist)
    return totalDist

def emd(x,y):
    '''Return emd between histograms defined by lists of floats x and y.

    Args:
    x,y -- lists of floats'''
    totalValues = sorted(list(set().union(x,y))) #Union of values in x and y
    x_dist = gen_hist(totalValues,sorted(x)) #histogram for x on totalValues
    y_dist = gen_hist(totalValues,sorted(y)) #histogram for y on totalValues
    return emdLinear(x_dist,y_dist,totalValues) #compute emd between histograms

def l2_exp_weight(x,y):
    '''Returns sum of squared differences between ith entries of x and y
    weighted exponentially by 1/e^i

    Args:
    x,y -- lists of floats 

    '''
    n = min(len(x),len(y))
    distance = 0
    for i in range(n):
        weight = np.exp(-1*i)
        distance = distance+(((x[i]-y[i])**2)*weight)
    return distance

def l2_lin_weight(x,y):
    '''Returns sum of squared differences between ith entries of x and y
    weighted linearly by 1/i

    Args:
    x,y -- lists of floats 

    '''
    n = min(len(x),len(y))
    distance = 0
    for i in range(n):
        weight = 1.0/float(i+1)
        distance = distance+(((x[i]-y[i])**2)*weight)
    return distance

def l2_weight(x,y,k=None):
    '''Returns sum of squared differences between ith entries of x and y
    weighted exponentially by 1/e^i

    Args:
    x,y -- lists of floats 
    k -- cutoff (default is None)

    '''
    n = min(len(x),len(y))
    distance = 0
    weight = 1.0
    if k is None:
        stop = n
    else:
        stop = k
    for i in range(stop):
        distance = distance+(((x[i]-y[i])**2)*weight)
    return distance


#LAPLACIAN SPECTRA###########

def sym_normalized_laplacian(A_input):
    '''Compute symmetric normalized laplacian of A (A may be fractional)

    Args:
    A_input -- adjacency matrix

    '''
    A = np.asarray(A_input)
    n = A.shape[0]
    d = A.sum(axis=0)
    d[d==0]=1
    sqrt_inv_D = np.diag(np.sqrt(1.0/d))
    L = np.eye(n) - np.matmul(np.matmul(sqrt_inv_D,A.astype(float)),sqrt_inv_D)
    return L

def specGap(A):
    '''Return second smallest eigenvalue of the 
    symmetirc normalized Laplacian of A

    Args:
    A -- adjacency matrix

    '''
    L = sym_normalized_laplacian(A.astype(float))
    w,v = np.linalg.eig(L)
    w = sorted(w)
    return np.real(w[1])

def smallest_nonzero_eig(A):
    '''Returns smallest non-zero eigenvalue of the 
    symmetric normalized Laplacian of A

    Args:
    A -- adjacency matrix

    '''
    L = sym_normalized_laplacian(A.astype(float))
    w,v = np.linalg.eig(L)
    return min(w[w>1e-12])

def spectrum(A):
    '''Return eigenvaleus of symmetric normalized laplacian of A

    Args:
    A -- adjacency matrix

    '''
    L = sym_normalized_laplacian(A.astype(float))
    w,v = np.linalg.eig(L)
    w = sorted(w)
    w = [np.real(x) for x in w]
    return w

def kthEigenVector(A,k):
    '''Compute eigenvector corresponding to the kth smallest eigenvalue in A

    Args:

    A -- matrix 
    k -- integer

    '''
    assert(k <= A.shape[0])
    w,v = np.linalg.eig(A)
    idx = np.argsort(w)
    v = v[:,idx]
    vec = v[:,k]
    return vec.real

def fiedlerVector(A):
    '''Compute eigenvector corresponding to the second smallest eigenvalue 
    of the symmetric normalized Laplacian of A

    Args:

    A -- matrix 

    '''
    L = sym_normalized_laplacian(A.astype(float))
    return kthEigenVector(L,1)

####CONDUCTANCE#####

def comp_complement(S,n):
    '''Return list of all the nodes in range(n) not in S

    Args:

    S -- list of nodes 
    n -- number of nodes (int)

    '''
    comp = []
    for i in range(n):
        if i not in S:
            comp.append(i)
    return comp

def compConductance(S,S_comp,A):
    '''Compute the proportion of weight of pairs crossing between S and S_comp
    to the minimum of sum of weights adjacenct to S/S_comp

    Args:
    S,S_comp -- bipartition of nodes 
    A -- (Fractional) adjacency matrix

    '''
    nodeSet = S + S_comp
    edges_S = A[S][:,nodeSet]
    edges_S_comp = A[S_comp][:,nodeSet]
    edges_cross = A[S][:,S_comp]
    c = float(edges_cross.sum())/float(min(edges_S.sum(), edges_S_comp.sum()))
    return c

def compMinConductanceCut(u,A):
    '''Return vertex set S = {v} where v in S if u[v]>T where T is chosen so S has lowest 
    conductance across all T in the set {u[v]}

    Args:
    u -- vector of floats
    A -- adjacency matrix

    '''
    n = len(u)
    idx = np.argsort(u)
    minConductance = 1
    clustering = [range(n),[]]
    for i in range(n-1):
        S = list(idx[:i+1])
        S_comp = list(idx[i+1:])
        c = compConductance(S,S_comp,A)
        if c < minConductance:
            minConductance = c 
            clustering = [S,S_comp]
    return clustering

def compMaxConductanceCut(u,A):
    '''Return vertex set S = {v} where v in S if u[v]>T where T is chosen so S has largest 
    conductance across all T in the set {u[v]}

    Args:
    u -- vector of floats
    A -- adjacency matrix

    '''
    n = len(u)
    idx = np.argsort(u)
    maxConductance = 0
    clustering = [range(n),[]]
    for i in range(n-1):
        S = list(idx[:i+1])
        S_comp = list(idx[i+1:])
        c = compConductance(S,S_comp,A)
        if c > maxConductance:
            maxConductance = c 
            clustering = [S,S_comp]
    return clustering

def total_variation_distance(x,y):
    '''Return total variatin distance between histograms x and y'''
    return .5*np.sum(np.abs(x-y))



###############################
##MISC ##########

def compNeighbors(edges, node_ixs, i):
    '''Return list of neighbors of the ith node

    Args:

    edges -- list of pairs partitioned by each node.
    node_ixs -- list of start index of the pairs in edges 
    belonging to each node

    '''
    N = len(node_ixs)
    if i == N-1:
        nbs = edges[node_ixs[i]:,1]
    else:
        nbs = edges[node_ixs[i]:node_ixs[i+1],1]
    return nbs

def randWalkMatrix(edges, node_ixs):
    '''Compute standard random walk tarnsition matrix.

    Args:

    edges -- list of pairs partitioned by each node.
    node_ixs -- list of start index of the pairs in edges 
    belonging to each node
    '''
    N = len(node_ixs)
    M = np.zeros([N,N])
    for i in range(N):
        nbs = compNeighbors(edges, node_ixs, i)
        num_nbs = len(nbs)
        for j in nbs:
            M[j,i]=1/float(num_nbs) #probability of moving from i to j
    return M

def vertexStateToEdgeStateTransition(P_v, edges, node_ixs):
    '''Compute edge transitions from first order node transitions in P_v

    Args:

    P_v -- vertex state to vertex state transition probabilities 
    edges -- list of pairs partitioned by each node.
    node_ixs -- list of start index of the pairs in edges 
    belonging to each node

    '''
    N = P_v.shape[0]
    M = len(edges)
    P = np.zeros([M,M])
    for edge_index in range(M):
        [i,j] = edges[edge_index]
        #independent of i, just need transition probabilities from P_v
        if j < N-1:
            neighbor_rows = range(node_ixs[j],node_ixs[j+1])
        else:
            neighbor_rows = range(node_ixs[j],len(edges))
        for index in neighbor_rows:
            #Probability of transitioning from edge (i,j) to edge (j,k)
            #is equivalent to transitioning from j to k
            [l,k] = edges[index,:]
            if l != j:
                print(ERROR)
            P[index,edge_index]=P_v[k,j]
    return P



def secondOrderRandWalkMatrix(edges, node_ixs, p = 1, q = 1):
    '''Compute edge transition matrices with the Node2vec walk using
    paramaters p,q

    Args:

    edges -- list of pairs partitioned by each node.
    node_ixs -- list of start index of the pairs in edges 
    belonging to each node
    p,q -- node2vec paramaters (default p = q = 1)

    '''
    N = len(node_ixs)
    M = len(edges)
    P = np.zeros([M,M])
    p = float(p)
    q = float(q)
    for edge_index in range(M):
        [i,j] = edges[edge_index] #fill at the column for having just traveled i->j
        prev_nbs = compNeighbors(edges, node_ixs,i)
        if j < N-1:
            neighbor_rows = range(node_ixs[j],node_ixs[j+1])
        else:
            neighbor_rows = range(node_ixs[j],len(edges))
        dist = []
        for index in neighbor_rows:
            [l,k] = edges[index,:]
            if l != j:
                print(ERROR)
            #compute probability traveling from j to k
            if k == i:
                dist.append(1.0/p)
            elif k in prev_nbs:
                dist.append(1.0)
            else:
                dist.append(1.0/q)
        dist = [x*(1.0/sum(dist)) for x in dist]
        if j < N-1:
            P[node_ixs[j]:node_ixs[j+1],edge_index]=dist
        else:
            P[node_ixs[j]:,edge_index]=dist
    return P


def solveStationary( A ):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    #I-A
    a = np.eye( n ) - A
    #Add vector for sum to 1
    a = np.vstack( (a.T, np.ones( n )) )
    #Should add up to zero and 1 for the probability vector
    b = np.matrix( [0] * n + [ 1 ] ).T
    #Solve for least squares solution
    s = np.linalg.lstsq( a, b, rcond=-1)[0]
    return s

def random_walk_w_matrix(N, rwlen, P, n_walks, p):
    '''Array of n_walks
    random walks of length rwlen using transition matrix P with stationary distribution 
    p on graph with N nodes 

    Args:

    N -- int, number of nodes 
    rwlen -- int, number of  nodes on walk
    P -- transition matrix, ith column is transition from node i 
    n_walks -- int, number of walks 
    p -- stationary distribution '''
    walk = []
    for t in range(n_walks):
        i = np.random.choice(N,p=p) #start node from stationary distribution
        walk.append(i)
        for k in range(rwlen-1):
            p_ = list(P[:,i])
            total = sum(p_)
            p_ = [float(x)/total for x in p_]
            p_ = [0 if x < 0 else x for x in p_] #standardize column of P
            i = np.random.choice(N, p=p_) #next node in walk
            walk.append(i)
    return np.array(walk)

def gen_single_node(N,P, p, cur_node=None):
    '''Sample single node from stationary distribution p if cur_node is None, 
    else walk from cur_node using transition probability matrix P

    Args:

    N -- int, number of nodes 
    P -- transtition matrix, ith column is transition from node i 
    p -- stationary distribution 
    cur_node -- current node of walk, default is None.

    '''
    if cur_node == None:
        cur_node = np.random.choice(N,p=p)
        return cur_node
    i = np.random.choice(N, p=P[:,cur_node])
    return i

def gnp(X):
    '''Sample symmetric binary matrix G using the upper triangle of probabilities in X'''
    X = np.triu(X)
    G = np.random.binomial(1,p=X)
    G = G+G.T
    return G


def normalize_mat(E, self_loops = True):
    '''Return normalized copy of E so that sum of entries
    is equal to 1 

    Args:
    E -- matrix 
    self-loops -- Boolean, if False diagonal is set to zero before normalization.
        default is True'''
    e = np.copy(E)
    if not self_loops:
        np.fill_diagonal(e, 0)
    e = (e / np.sum(e))
    return e

def normMatrix_wsum(Mc,m,m_exact = True):
    '''Return normalized copy of E so that sum of entries
    is equal to m and no entry is larger than 1.

    Args:
    Mc -- matrix 
    m -- int 
    m_exact -- Boolean, if False the normalized version of the 
        matrix with sum equal to 1 is multiplied by m and and all
        entries above 0 are set to 1. If True, then entries that are not
        eqaul to 1 continue to be rounded up until there are no entries 
        left or matrix sum is equal to m '''
    M = np.array(copy.deepcopy(Mc))
    np.fill_diagonal(M,0)
    total = M.sum()
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            M[i,j] = M[i,j]/total 
    M = M*m
    num_above = M[M>1].shape[0]
    M[M>1]=1
    if not m_exact:
        return M
    #increase all edges that are not maxed out at 1
    while M.sum() < m-.001:
        amount_left = m-M[M>=1].sum() #what we want edges below 1 to add up to
        total = M[M<1].sum() #current total
        if (len(M[M<1])-len(M[M<=0])) == 0: #if all entries less than 1 are 0, break
            break
        for i in range(n):
            for j in range(n):
                if M[i,j] < 1:
                    M[i,j] = (M[i,j]/total)*amount_left #normalized weight * desired 
        M[M>1]=1
    return M

def entropy_m(X):
    '''Return matrix of the entropies of 
    bernoulli random variables in X 

    Args:

    X -- matrix of entries in [0,1]'''
    n = X.shape[0]
    H = np.zeros(X.shape)
    for i in range(n):
        for j in range(n):
            H[i][j] = stat.entropy([X[i][j],1-X[i][j]])
    return H

def entropy_m_sum(X):
    '''Return sum of the entropies of the bernoulli random variables in X 

    Args:

    X -- matrix of entries in [0,1]'''

    H = entropy_m(X)
    return H.sum()


def auc_p(X,A,valid):
    '''Return roc_auc_score and ap scores for how well 
    X predicts A for pairs in valid 

    Args:

    X -- prediction score matrix
    A -- true score matrix 
    valid -- list of pairs 

    '''
    trueScores = []
    predScores = []
    for [u,v] in valid:
        trueScores.append(A[int(u)][int(v)])
        predScores.append(X[int(u)][int(v)])
    auc = roc_auc_score(trueScores,predScores)
    ap = average_precision_score(trueScores,predScores)
    return auc,ap


#########################################

def second_order_walk(edges, node_ixs, rwlen, p=1, q=1, n_walks=1):

    '''Array of n_walks
    random walks of length rwlen using node2vec walk

    Args:

    edges -- list of pairs partitioned by each node.
    node_ixs -- list of start index of the pairs in edges 
    belonging to each node 
    rwlen -- int, number of  nodes on walk
    p,q -- node2vec paramaters
    n_walks -- number of walks 
     '''


    N=len(node_ixs)
    
    walk = []
    prev_nbs = None
    for w in range(n_walks):
        source_node = np.random.choice(N)
        walk.append(source_node)
        for it in range(rwlen-1):
            
            if walk[-1] == N-1:
                nbs = edges[node_ixs[walk[-1]]:,1]
            else:
                nbs = edges[node_ixs[walk[-1]]:node_ixs[walk[-1]+1],1]
                
            if it == 0:
                walk.append(np.random.choice(nbs))
                prev_nbs = set(nbs)
                continue

            is_dist_1 = []
            for n in nbs:
                is_dist_1.append(int(n in set(prev_nbs)))

            is_dist_1_np = np.array(is_dist_1)
            is_dist_0 = nbs == walk[-1]
            is_dist_2 = 1 - is_dist_1_np - is_dist_0

            alpha_pq = is_dist_0 / float(p) + is_dist_1_np + is_dist_2/float(q)
            alpha_pq_norm = alpha_pq/np.sum(alpha_pq)
            rdm_num = np.random.rand()
            cumsum = np.cumsum(alpha_pq_norm)
            nxt = nbs[np.sum(1-(cumsum > rdm_num))]
            walk.append(nxt)
            prev_nbs = set(nbs)
    return np.array(walk)

class RandomWalker:
    """
    Helper class to generate random walks on the input adjacency matrix.
    """
    def __init__(self, adj, rw_len=16, p=1, q=1, batch_size=128, comp_combo = True):
        self.adj = adj
        self.rw_len = rw_len
        self.p = float(p)
        self.q = float(q)
        self.edges = np.array(self.adj.nonzero()).T
        #Starting index of set of edges in edges adjacent to each node
        self.node_ixs = np.unique(self.edges[:, 0], return_index=True)[1]
        self.batch_size = batch_size
        self.N = len(self.node_ixs)
        self.A = adj.todense()
        self.A_wsl = sp.csr_matrix(self.A + np.eye(self.N))
        self.edges_wsl = np.array(self.A_wsl.nonzero()).T
        self.node_ixs_wsl = np.unique(self.edges_wsl[:, 0], return_index=True)[1]
        numEdges = self.A.sum()
        D = self.A.sum(axis=0).tolist()[0]
        self.RW = randWalkMatrix(self.edges, self.node_ixs)
        s_temp = solveStationary(np.transpose(self.RW))
        self.s = [x.tolist()[0][0] for x in s_temp]
        self.current_node = None
        #self.SORW = secondOrderRandWalkMatrix(self.edges, self.node_ixs, self.p, self.q)
        
        #print(self.s)
        #self.so = solveStationary(np.transpose(self.SORW))
        #self.so = [x.tolist()[0][0] for x in so_test]
        if comp_combo:
            self.P, self.fmm_time = fast_mix_transition_matrix_slem(self.A)
            self.Combo = .5*self.P + .5*self.RW 
            combo_stationary =solveStationary(np.transpose(self.Combo))
            self.combo_stationary = [x.tolist()[0][0] for x in combo_stationary]
            #C_edge = vertexStateToEdgeStateTransition(self.Combo,self.edges_wsl,self.node_ixs_wsl)
            #self.combo_edge_stationary = solveStationary(np.transpose(C_edge))

    def walk(self):
        while True:
            yield random_walk_w_matrix(self.N, self.rw_len, self.RW, self.batch_size, self.s).reshape([-1, self.rw_len])

    def combo_walk(self):
        while True:
            yield random_walk_w_matrix(self.N, self.rw_len, self.Combo, self.batch_size, self.combo_stationary).reshape([-1, self.rw_len])

    def rand_vertex_from_subset(self,S):
        y = self.A[S]
        deg_s = np.asarray(y).sum(axis=1)
        total_deg_s = deg_s.sum()
        p = [deg/total_deg_s for deg in deg_s]
        i = np.random.choice(range(len(S)),p=p)
        return S[i]
        

    def set_current(self, S=None):
        if S is None:
            self.current_node = self.rand_vertex_from_subset(range(self.N))
        else:
            self.current_node = self.rand_vertex_from_subset(S)

    
    def walk_single(self):
        while True:
            self.current_node = gen_single_node(self.N,self.RW,self.s,self.current_node)
            yield self.current_node


'''Benchmark generators'''

def fitSBM(A,num_clusters):
    ''' Return probabilistic adjacency matrix for SBM on graph
    G defined by A with num_clusters

    Args:

    A -- adjacency matrix 
    num_clusters -- int '''
    G = nx.from_numpy_matrix(A)
    #fit the model 
    clustering = SpectralClustering(n_clusters=num_clusters,affinity='precomputed').fit(A)
    n = len(clustering.labels_) #ith entry of clustering.labels_ is cluster of node i
    total = range(n)
    clusters = [[] for x in range(num_clusters)]
    S = np.zeros((n,n))
    for i in total:
        clusters[clustering.labels_[i]].append(i) #append i to the list coresponding to its cluster
    intra_cluster = np.zeros((num_clusters,num_clusters))
    inter_cluster = np.zeros((num_clusters))
    for i in range(num_clusters): #compute densities
        Y = A[np.ix_(clusters[i],clusters[i])] 
        size = float(len(clusters[i]))
        inter_cluster[i] = float(Y.sum())/(size*(size-1))
        for j in range(i+1,num_clusters):
            Y = A[np.ix_(clusters[i],clusters[j])]
            density = float(Y.sum())/(size*len(clusters[j]))
            intra_cluster[i,j] = density
            intra_cluster[j,i]= density
    for i in range(n):
        for j in range(i+1,n):
            if clustering.labels_[i]==clustering.labels_[j]:
                S[i,j] = inter_cluster[clustering.labels_[i]]
            else:
                S[i,j] = intra_cluster[clustering.labels_[i],clustering.labels_[j]]
    return S+S.T



##########################
'''@inproceedings{DBLP:conf/icml/BojchevskiSZG18,
  author    = {Aleksandar Bojchevski and
               Oleksandr Shchur and
               Daniel Z{\"{u}}gner and
               Stephan G{\"{u}}nnemann},
  title     = {NetGAN: Generating Graphs via Random Walks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning,
               {ICML} 2018, Stockholmsm{\"{a}}ssan, Stockholm, Sweden, July
               10-15, 2018},
  pages     = {609--618},
  year      = {2018},
}'''


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)['arr_0'].item()
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False

    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges

    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = np.array(list(nx.maximal_matching(nx.DiGraph(A))))
                not_in_cover = np.array(list(set(range(N)).difference(hold_edges.flatten())))

                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                min_size = hold_edges.shape[0] + len(not_in_cover)
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))

                d_nic = d[not_in_cover]

                hold_edges_d1 = np.column_stack((not_in_cover[d_nic > 0],
                                                 np.row_stack(map(np.random.choice,
                                                                  A[not_in_cover[d_nic > 0]].tolil().rows))))

                if np.any(d_nic == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, not_in_cover[d_nic == 0]].T.tolil().rows)),
                                                     not_in_cover[d_nic == 0]))
                    hold_edges = np.row_stack((hold_edges, hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = np.row_stack((hold_edges, hold_edges_d1))

            else:
                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        test_zeros = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        test_zeros = np.row_stack(test_zeros)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """

    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat

def edge_overlap(A, B):
    """
    Compute edge overlap between input graphs A and B, i.e. how many edges in A are also present in graph B. Assumes
    that both graphs contain the same number of edges.

    Parameters
    ----------
    A: sparse matrix or np.array of shape (N,N).
       First input adjacency matrix.
    B: sparse matrix or np.array of shape (N,N).
       Second input adjacency matrix.

    Returns
    -------
    float, the edge overlap.
    """

    return ((A == B) & (A == 1)).sum()


def graph_from_scores(scores, n_edges):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.

    Parameters
    ----------
    scores: np.array of shape (N,N)
            The input transition scores.
    n_edges: int
             The desired number of edges in the target graph.

    Returns
    -------
    target_g: symmettic binary sparse matrix of shape (N,N)
              The assembled graph.

    """

    if  len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0

    target_g = np.zeros(scores.shape) # initialize target graph
    scores_int = scores.toarray().copy() # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)   # The row sum over the scores.

    N = scores.shape[0]

    for n in np.random.choice(N, replace=False, size=N): # Iterate the nodes in random order

        row = scores_int[n,:].copy()
        if row.sum() == 0:
            continue

        probs = row / row.sum()

        target = np.random.choice(N, p=probs)
        target_g[n, target] = 1
        target_g[target, n] = 1


    diff = np.round((n_edges - target_g.sum())/2)
    if diff > 0:

        triu = np.triu(scores_int)
        triu[target_g > 0] = 0
        triu[np.diag_indices_from(scores_int)] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_int)
        extra_edges = np.random.choice(triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff))

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    target_g = symmetric(target_g)
    return target_g


def symmetric(directed_adjacency, clip_to_one=True):
    """
    Symmetrize the input adjacency matrix.
    Parameters
    ----------
    directed_adjacency: sparse matrix or np.array of shape (N,N)
                        Input adjacency matrix.
    clip_to_one: bool, default: True
                 Whether the output should be binarized (i.e. clipped to 1)

    Returns
    -------
    A_symmetric: sparse matrix or np.array of the same shape as the input
                 Symmetrized adjacency matrix.

    """

    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric




def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)







def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er

def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:,None].dot(counts[None,:])
        if normalize:
            blocks_outer = np.multiply(block, 1/blocks_outer)
        return blocks_outer
    
    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1-np.eye(in_blocks.shape[0])).mean() 
    return diag_mean, offdiag_mean

def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.
          
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """

    A = A_in.copy()

    assert ((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]
    
    if Z_obs is not None:
        # inter- and intra-community density
        intra, inter = statistics_cluster_props(A, Z_obs)
        statistics['intra_community_density'] = intra
        statistics['inter_community_density'] = inter
      
    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics
