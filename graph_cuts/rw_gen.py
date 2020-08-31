import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

import numpy as np
import scipy.sparse as sp
import random
import copy
from sklearn.cluster import spectral_clustering
import csv
import itertools
from collections import defaultdict
import math
import networkx as nx
import random
from datetime import datetime
import json
import time
import quadprog
import scipy.special as ss
import shutil
import cut_walk_gen_utils

'''TESTS'''

def gen_indep_walks_barbell():
	G = nx.barbell_graph(50,0)
	A = nx.adjacency_matrix(G).todense()
	for i in range(11,21):
		for j in range(80,90):
			A[i,j] = 1
			A[j,i] = 1
	walker1, walker2 = gen_walkers(A)
	walker1.set_current([5])
	walker2.set_current([55])
	(w1,w2,S,S_comp) = gen_indep_walks(walker1, walker2)
	print(w1)
	print(w2)
	print(S)
	print(S_comp)


def gen_votes_cluster():
	G = nx.barbell_graph(50,0)
	A = nx.adjacency_matrix(G).todense()
	for i in range(11,21):
		for j in range(80,90):
			A[i,j] = 1
			A[j,i] = 1
	walker1, walker2 = gen_walkers(A)
	walker1.set_current([5])
	walker2.set_current([55])
	(w1,w2,S,S_comp) = gen_indep_walks(walker1, walker2)
	votes, walks = gen_votes(walker1,S,S_comp,A.shape[0],10)
	print(S)
	print(S_comp)
	print(votes[10])
	print(votes[30])
	print(votes[60])
	print(votes[90])
	for u in [10,30,60,90]:
		if ((votes[u][0]+votes[u][1]) == 0):
			print('{} is seed node'.format(u))
			print((u in S) or (u in S_comp))

def matrix_update_barbell():
	G = nx.barbell_graph(5,0)
	A = nx.adjacency_matrix(G).todense()
	S = set([0,1,2,3,4])
	S_comp = set([5,6,7,8,9])
	batch_size = 1
	F = np.zeros((10,10))
	params = {'bs':1,'walk_dist_weight':True,'zero_cross':True}
	num_iters = [5]
	walks = [[[2,4,5],[8,7,6],[3,1,2,3,4,5,6]]]
	stats = defaultdict(list)
	frequency_map = {}
	truth_spec = utils.spectrum(A)
	batch_num = 0
	F, stats  = matrix_update(A, S, S_comp, params, F, num_iters, walks, stats, batch_num, truth_spec)
	print(F)



'''UTILS'''

def gen_walkers(A_truth):
	'''Initializing parameters for walker constructors'''
	_A_obs = sp.csr_matrix(A_truth)
	_A_obs[_A_obs > 1] = 1
	lcc = utils.largest_connected_components(_A_obs)
	_A_obs = _A_obs[lcc,:][:,lcc]
	walker1 = utils.RandomWalker(_A_obs, comp_combo = False)
	walker2 = utils.RandomWalker(_A_obs, comp_combo = False)
	return walker1, walker2


'''CONFIGS'''


def gen_walk_config(alg):
	config = {}
	config['scale']=False
	config['init_method'] = 'lazy'
	config['eig_init']=False
	config['bs']=1
	config['num_cut_init_walks']=20
	config['zero_cross']=False
	config['walk_dist_weight']=False
	comp = False
	if alg == 'bs_1':
		comp = True 
	if alg == 'bs_1+zc':
		comp = True 
		config['zero_cross'] = True
	if alg == 'bs_1+wdw':
		comp = True 
		config['walk_dist_weight'] = True
	if alg == 'bs_1+zc+wdw':
		comp = True 
		config['zero_cross'] = True
		config['walk_dist_weight'] = True
	if alg == 'bs_5':
		comp = True 
		config['bs'] = 5
	if alg == 'bs_5+zc':
		comp = True 
		config['bs'] = 5
		config['zero_cross'] = True
	if alg == 'bs_5+wdw':
		comp = True 
		config['bs'] = 5
		config['walk_dist_weight'] = True
	if alg == 'bs_5+zc+wdw':
		comp = True 
		config['bs'] = 5
		config['zero_cross'] = True
		config['walk_dist_weight'] = True
	if alg == 'bs_10':
		comp = True 
		config['bs'] = 10
	if alg == 'bs_10+zc':
		comp = True 
		config['bs'] = 10
		config['zero_cross'] = True
	if alg == 'bs_10+wdw':
		comp = True 
		config['bs'] = 10
		config['walk_dist_weight'] = True
	if alg == 'bs_10+zc+wdw':
		comp = True 
		config['bs'] = 10
		config['zero_cross'] = True
		config['walk_dist_weight'] = True
	if alg == 'bs_20':
		comp = True 
		config['bs'] = 20
	if alg == 'bs_20+zc':
		comp = True 
		config['bs'] = 20
		config['zero_cross'] = True
	if alg == 'bs_20+wdw':
		comp = True 
		config['bs'] = 20
		config['walk_dist_weight'] = True
	if alg == 'bs_20+zc+wdw':
		comp = True 
		config['bs'] = 20
		config['zero_cross'] = True
		config['walk_dist_weight'] = True
	if alg == 'bs_40':
		comp = True 
		config['bs'] = 40
	if alg == 'bs_40+zc':
		comp = True 
		config['bs'] = 40
		config['zero_cross'] = True
	if alg == 'bs_40+wdw':
		comp = True 
		config['bs'] = 40
		config['walk_dist_weight'] = True
	if alg == 'bs_40+zc+wdw':
		comp = True 
		config['bs'] = 40
		config['zero_cross'] = True
		config['walk_dist_weight'] = True
	return comp, config


'''RANDOM WALK GEN'''

def count_opp_walks(walker1,walker2,k,n):
	'''For each node pair (u,v), count the number of times (u,v) appear
	in opposite walks started from different points with walks terminated
	once they collide our of k total walk pairs

	Return counts 

	Args:
	walker1, walker2 : independent generators for walking on template graph 
	k: number of pairs of walks 
	n: number of nodes'''
	counts = np.zeros((n,n))
	for i in range(k):
		#initialize walks in different places
		walker1.set_current(range(n))
		walker2.set_current(range(n))
		while walker2.current_node == walker1.current_node: 
			walker2.set_current(range(n))
		(w1,w2) = gen_indep_walks(walker1, walker2) #walks terminated once they collide.
		for u in w1:
			for v in w2:
				counts[u,v] = counts[u,v]+1
				counts[v,u] = counts[v,u]+1
	return counts


def gen_indep_walks(walker1, walker2):
	'''Form two subsets s1,s2 of vertices in G by starting
	two walks indpendently. All vertices on the first walk are 
	in s1, all vertices in the second walk are in s2. Break once
	the walks intersect.
	Assumes walkers have two different current nodes.

	Returns: Two lists of vertices hit by each walk 

	Args:
	walker1, walker2: independent generators for walking on template graph
	'''
	walk1 = walker1.walk_single
	walk2 = walker2.walk_single
	w1 = [walker1.current_node]
	w2 = [walker2.current_node]
	s1 = set(w1)
	s2 = set(w2)
	while True:
		assert(len(s1.intersection(s2))==0) 
		i1 = next(walk1())
		i2 = next(walk2())
		if i1 == i2: #return if the walks intersect at the same point
			return(w1,w2,s1,s2)
		if i1 in s2: #return if walk1 hits node hit by walk2
			return (w1,w2,s1,s2)
		if i2 in s1: #return if walk2 hits node hi by walk1
			return (w1,w2,s1,s2)
		s1.add(i1) #else, add nodes visited to the walks and s1,s2
		s2.add(i2)
		w1.append(i1)
		w2.append(i2)


def initialize_cuts(n,A,walker1,walker2,init_method,num_cut_init_walks=100):
	'''Initializes independnet walks according to init_method runs walks until they intersect 

	Returns two lists of vertices hit by each indpendent walk, vertex sets hit will be disjoint

	Args:
	n - int, number of vertices 
	A - adjacency matrix of template graph 
	walker1, walker2 - generators for walking on template graph 
	init_method - how to initialize start vertices of walks, default is weighted by degree 
	num_cut_init_walks - number of walks if initializing by counts (default = 100) '''
	if init_method == 'eig':
		'''Initialize walks on opposite sides of the convex combination of 
		the eigenvectors corresponding to the five smallest non-zero eigenvaleus'''
		L = utils.sym_normalized_laplacian(A)
		top_eigenvectors = [utils.kthEigenVector(L,i) for i in range(1,6)]
		v = random_convex_comb(top_eigenvectors)
		c, c_comp = utils.compMinConductanceCut(v,A)
		walker1.set_current(c)
		walker2.set_current(c_comp)
	elif init_method == 'cow':
		'''Count the number of times node pairs are on opposite walks'''
		node_pair_count = count_opp_walks(walker1,walker2,num_cut_init_walks,A.shape[0])
		'''Normalize the counts'''
		node_pair_prob = utils.normalize_mat(node_pair_count).flatten()
		'''Use normalized counts as a distribution over pairs'''
		pairs = np.transpose([np.tile(range(n), n), np.repeat(range(n), n)])
		pair_num = np.random.choice(len(pairs), 1, p=node_pair_prob)
		'''Pick pair'''
		[u,v] = pairs[pair_num[0]]
		assert(u!=v)
		walker1.set_current([u])
		walker2.set_current([v])
	else:
		'''Initialize walks randomly from different points weighted by degree'''
		walker1.set_current(range(n))
		walker2.set_current(range(n))
		while walker2.current_node == walker1.current_node:
			walker2.set_current(list(range(n)))
	assert(walker2.current_node != walker1.current_node)
	return gen_indep_walks(walker1, walker2)

def gen_votes(walker,S,S_comp,n,batch_size = 1):
	'''Run a batch of walk algorithm iterations in which walks are run until walk hits a node 
	in one of the disjoint sets of seed nodes. Keep track of the number of times (votes) each non-seed walk 
	node hits each of the disjiont sets. 

	Returns votes and a list of walks for each walk algorithm iteration

	Args: 

	walker: generator for walking on template graph 
	S, S_comp: disjoint sets of seed nodes 
	n: number of vertices 
	batch_size: number of walk algorithm iterations to run '''
	votes = np.zeros((n,2))
	total_walks = [[]]*batch_size
	partial_all = S.union(S_comp)
	num_in_partial_all = len(partial_all)
	placed_all_rounds = copy.deepcopy(partial_all)
	max_walks = 1000
	for batch_num in range(batch_size):
		start = time.time()
		walks_in_round = []
		unseen_list = cut_walk_gen_utils.complement([S,S_comp],n) #compute nodes not in seed sets
		unseen = set(unseen_list)
		num_placed = num_in_partial_all
		walker.set_current(unseen_list) #set walker at unseen node
		walk_gen = walker.walk_single
		w = [walker.current_node]
		s = set(w)
		seen = set()
		while num_placed < n and len(walks_in_round) < max_walks: 
			assert(len(partial_all) + len(seen) + len(unseen) == n)
			i = next(walk_gen())
			placed_all_rounds.add(i)
			if i in S: #walk until we hit a seed node
				for u in w:
					votes[u][0] = votes[u][0]+1
			if i in S_comp:
				assert(i not in S)
				for u in w:
					votes[u][1] = votes[u][1]+1
			if i in partial_all: #if we hit a seed node, record walk
				seen_before = len(seen)
				num_new = 0
				for v in s:
					seen.add(v)
					if v in unseen:
						unseen.remove(v)
						num_placed = num_placed + 1
						num_new = num_new + 1
				assert(len(seen) == (seen_before + num_new))
				walks_in_round.append(w)
				if num_placed < n: #if still nodes that have not been seen, start a new walk
					walker.set_current(list(unseen))
					walk_gen = walker.walk_single
					w = [walker.current_node]
					s = set(w)
			else: #update walk and set
				s.add(i)
				w.append(i)
		assert((num_placed == n) or (len(walks_in_round) == max_walks))
		total_walks[batch_num] = walks_in_round
	return votes, total_walks


def gen_cut_from_walks(A,walker1,walker2,walk_params,init_size_thresh = 5):
	'''Generate cut using independent random walks. 

	Returns bipartition of vertices (S, S_comp) and list of walks. 

	Args:

	A - Adjacency matrix of template graph.
	walker1, walker2 - indpendent genertaors to walk on template graph.
	walk_params - hyperparamater map 
	init_size_thresh - minimum number of nodes to be seen by seed walks (default is 5)
	'''
	init_method = walk_params['init_method']
	batch_size = walk_params['bs']
	num_cut_init_walks = walk_params['num_cut_init_walks']
	n = A.shape[0]
	S = set()
	S_comp = set()

	while min(len(S),len(S_comp)) < init_size_thresh: #seed walks
		(w1,w2,S,S_comp) = initialize_cuts(n,A,walker1,walker2,init_method,num_cut_init_walks)
	walks = []
	nodes_to_label = cut_walk_gen_utils.complement([S,S_comp],n)
	assert((len(S)+len(S_comp)+len(nodes_to_label))==n)
	
	if len(nodes_to_label)>0: #if nodes left to label, vote by generating batch of votes
		assert(len(S.intersection(S_comp))==0)
		votes, walks = gen_votes(walker1,S,S_comp,n,batch_size)
		for u in nodes_to_label:
			if votes[u][0] > votes[u][1]:
				S.add(u)
			else:
				S_comp.add(u)

	assert(len(S) + len(S_comp) == n)
	assert(len(S.intersection(S_comp))==0)
	if len(nodes_to_label) == 0:
		walks = [[w1]+[w2]]
	else:
		walks = [[w1]+[w2]+walks[i] for i in range(batch_size)]
	return (S,S_comp,walks)

def num_nodes_in_cut(S,S_comp,walk):
	'''Return number of nodes seen by walk before crossing cut (S, S_comp)

	Args: 

	S, S_comp - Disjoint sets of vertices (cut)
	walk - list of vertices '''

	if walk[0] in S:
		walk_sign = 1
	else:
		walk_sign = -1
	walk_len = len(walk)
	exited_cut = False
	for i in range(walk_len):
		v_i = walk[i]
		if v_i in S and walk_sign == -1:
			exited_cut = True
			break
		elif v_i in S_comp and walk_sign == 1:
			exited_cut = True
			break
	if exited_cut:
		return i
	else:
		return walk_len

def comp_updates_node_dist(walk, distance, walk_dist_weight,weights, x, y):
	'''Compute weights for pairs of nodes in walk distance number of steps apart

	Return update lists of node pairs and weights

	Args:

	walk - list of vertices 
	distance - int for number of steps between nodes
	walk_dist_weight - Boolean flag 
	weights - list of weights 
	x - list of first nodes in pairs 
	y - list of second nodes in pairs '''

	first_nodes = walk[:-distance]
	second_nodes = walk[distance:]
	weights_new = np.ones(len(first_nodes)*2) #symmetric, need updates in both directions
	if walk_dist_weight:
		weights_new = weights_new*(1.0/float(distance)) #weight inversely proportional to distance of nodes
	weights = np.append(weights,weights_new)
	x = np.append(x,first_nodes)
	x = np.append(x,second_nodes)
	y = np.append(y,second_nodes)
	y = np.append(y,first_nodes)
	return weights,x,y


def update_F(walks,F,walk_dist_weight,cut_disc,S,S_comp):
	'''Using cuts and walk, update frequency matrix F 
	using counts of node pairs on the walks and cut (S, S_comp) 

	Return updated frequency matrix F

	Args:

	walks - list of walks 
	F - frequency matrix 
	walk_dist_weight - Boolean flag 
	cut_disc - Boolean flag 
	S, S_comp - bipartition of vertices (cut) '''

	num_nodes = F.shape[0]
	assert(len(S)+len(S_comp)==num_nodes)
	updates = np.zeros_like(F) #will be added to F in the end
	weights = np.array([]) #entries of updates
	x = np.array([]) #coordinates of updates
	y = np.array([]) #coordinates of updates
	max_dist = 20 #cap for number of steps between nodes for which to add weight
	for walk in walks:
		if cut_disc: #cap is at most number of steps between first node and last node in cut
			k = min(max_dist,num_nodes_in_cut(S,S_comp,walk))
		else: #cap is at most length of walk
			k = min(max_dist,len(walk))
		for dist in range(1,k): #comp update for pairs of nodes dist steps apart
			weights, x, y = comp_updates_node_dist(walk, dist, walk_dist_weight,weights, x, y)
	updates = sp.coo_matrix((weights,(x,y)),shape=(num_nodes,num_nodes)).toarray() #make matrix
	np.fill_diagonal(updates,0) #remove self loops
	return F + updates

def matrix_update(A, S, S_comp, walk_params, F, num_iters, walks, stats, batch_num, truth_spec):
	'''Given cut S,S_comp and list of walks used to find S,S_comp, update F 
	according to walk_dist_weight and zero_cross params.
	
	Return updated F and
	stats which maps statistic names to a list of statistic values on updated frequency matrices
	
	Args:
	A - Adjacency matrix of template graph 
	S, S_comp - cut
	walk_params - map of paramaters for update
	F - current frequency matrix 
	num_iters - number of walk algorithm iterations 
	walks - list of walks 
	stats - map of statistic names to values 
	batch_num - number of the batch 
	truth_spec - spectrum of symmetric normalized Laplacian of A 
	'''

	batch_size = walk_params['bs']
	walk_dist_weight = walk_params['walk_dist_weight']
	zero_cross = walk_params['zero_cross']
	for j in range(batch_size): #Updates F for the jth walk algorithm iteration in batch
		start = time.time()
		if (len(walks) == 1): #Batch labeled all nodes with the seed walks
			F = update_F(walks[0],F,walk_dist_weight,zero_cross,S,S_comp)
			walk_num = 0
			final_walk = 0
		elif (len(walks) > 1): #Batch computed a batch_size number of walks
			assert(len(walks)==batch_size)
			F = update_F(walks[j],F,walk_dist_weight,zero_cross,S,S_comp)
			walk_num = j
			final_walk = batch_size-1					
		end = time.time()
		prev_sum = F.sum() #None of this should change F
		cut_conductance = utils.compConductance(list(S),list(S_comp),A)
		stats['conductance'].append(cut_conductance)
		stats['cuts'].append(list(S))
		stats['first_walk_nodes'].append([walks[walk_num][0],walks[walk_num][1]])
		walk_lens = [len(walk) for walk in walks[walk_num]]
		num_transitions = [len(walk)-1 for walk in walks[walk_num]]
		if (j==final_walk) and (batch_num%2 == 1): #Record statistics every other batch
			if ((batch_num*batch_size)+j) not in num_iters: 
				normF = utils.normMatrix_wsum(F,A.sum())
			spec_check = utils.spectrum(normF)
			l2_lin_check = utils.l2_lin_weight(spec_check,truth_spec)
			entropy_check = np.mean(utils.entropy_m(normF))
			stats['checkpoints'].append((batch_num+1)*batch_size)
			stats['batch_size'].append(batch_num+1)
			stats['l2_lins'].append(l2_lin_check)
			stats['num_expect_edges'].append(normF.sum())	
			stats['entropy_list'].append(entropy_check)	
			stats['spectra'].append(spec_check)
			stats['num_walks'].append(len(walks[j]))
			stats['mean_walk_len'].append(np.mean(walk_lens))
			stats['total_transitions'].append(np.sum(num_transitions))
			stats['time_to_update'].append(end-start)
		assert(F.sum() == prev_sum) #F should not be changed
		if len(walks) == 1:
			break
	return F, stats


def gen_freq_from_walks(A,walker1,walker2,num_iters,walk_params,data_path=None,freq_path = None):
	'''Generate probabilistic adjacency matrix using random walk generation
	Return map of walk algorithm iterations to number of seconds

	Args:
	A - adjacency matrix of target graph 
	walker1, walker2 - independent generators to walk along target graph
	num_iters - list of number of walk algorithm iterations to run before writing scaled 
		probabilistic adjacency matrix 
	walk_params - hyperparamater map 
	data_path - path to write statistics of cuts found 
	freq_path - paths for writing probabilistc adjacency matrices 

	'''
	start = time.time()
	times = {}
	truth_spec = utils.spectrum(A)
	F = np.zeros((A.shape[0],A.shape[0])) #Initialize frequency matrix
	stats = defaultdict(list)
	num_rounds = [int(x/walk_params['bs']) for x in num_iters] #number of rounds is number of iterations divided by batch size
	max_rounds = max(num_rounds) 
	for i in range(max_rounds):
		(S,S_comp,walks) = gen_cut_from_walks(A,walker1,walker2,walk_params)
		assert(len(S.intersection(S_comp)) == 0)
		F, stats = matrix_update(A, S, S_comp, walk_params, F, num_iters, walks, stats, i, truth_spec)
		if ((i+1)*walk_params['bs']) in num_iters: #record time
			if freq_path != None:
				sF = utils.normMatrix_wsum(F,A.sum())
				np.savetxt(freq_path+'num_iters{}.txt'.format((i+1)*walk_params['bs']),sF)
			times[(i+1)*walk_params['bs']] = time.time()-start
	if data_path !=None: #recording data
		with open(data_path+"cuts.csv","w") as f:
		    wr = csv.writer(f)
		    wr.writerows(stats['cuts'])
		f.close()
		with open(data_path+"first_walk_nodes.csv","w") as f:
		    wr = csv.writer(f)
		    wr.writerows(stats['first_walk_nodes'])
		f.close()
		np.savetxt(data_path+'conductance.txt',stats['conductance'])
		np.savetxt(data_path+'checkpoints.txt',stats['checkpoints'])
		np.savetxt(data_path+'batch_size.txt',stats['batch_size'])
		np.savetxt(data_path+'l2_lin.txt',stats['l2_lins'])
		np.savetxt(data_path+'num_expect_edges.txt',stats['num_expect_edges'])
		np.savetxt(data_path+'entropy.txt',stats['entropy_list'])
		np.savetxt(data_path+'spectra.txt',stats['spectra'])
		np.savetxt(data_path+'num_walks.txt',stats['num_walks'])
		np.savetxt(data_path+'mean_walk_len.txt',stats['mean_walk_len'])
		np.savetxt(data_path+'total_transitions.txt',stats['total_transitions'])
		np.savetxt(data_path+'time_to_update.txt',stats['time_to_update'])
	return times



def rw_gen(A,hyperparams,num_iters,prob_adj_paths, data_paths):
	'''Compute probabilistic adjacency matrix for each hyperparamater set.
	Return map of hyperparamater set to map of walk algorithm iterations to number of seconds

	Args:

	A - Adjacency matrix of template graph 
	hyperparams - random walk generation hyperparamater sets 
	num_iters - map of hyperparamater set to 
		list of number of random walk algorithm iterations to run  
	prob_adj_paths - map of hyperparamater set to 
		path for writing the probabilistic adjacency matrices
	data_paths - map of hyperparamater set to 
		path for writing data on cuts sampled'''

	times = {}
	walker1, walker2 = gen_walkers(A)
	for y in hyperparams:
		comp, config = gen_walk_config(y) #dictionary of args for hyperparamaters
		if comp == True:
			prob_adj_path = prob_adj_paths[y]
			data_path = data_paths[y]
			num_iters_alg = num_iters[y]
			times[y] = gen_freq_from_walks(A,walker1,walker2,num_iters_alg,config,data_path,prob_adj_path)
		else:
			print(alg_missing)
	return times





	
