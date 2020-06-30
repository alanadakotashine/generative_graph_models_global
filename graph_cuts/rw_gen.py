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
	walker1, walker2 = gen_walkers(A)
	walker1.set_current([5])
	walker2.set_current([55])
	(w1,w2,S,S_comp) = gen_indep_walks(walker1, walker2)


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
	for walkset in walks:
		print(len(walkset))
	for u in [10,30,60,90]:
		if ((votes[u][0]+votes[u][1]) == 0):
			print('{} is seed node'.format(u))
			print((u in S) or (u in S_comp))



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
	once they collide.'''
	counts = np.zeros((n,n))
	for i in range(k):
		#initialize walks in different places
		walker1.set_current(range(n))
		walker2.set_current(range(n))
		while walker2.current_node == walker1.current_node:
			walker2.set_current(range(n))
		#walks terminated once they collide.
		(w1,w2) = gen_indep_walks(walker1, walker2)
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
	Assumes walkers have two different current nodes.'''
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
		#return if the walks intersect at the same point or hit
		#a node placed in s1 or s2
		if i1 == i2:
			return(w1,w2,s1,s2)
		if i1 in s2:
			return (w1,w2,s1,s2)
		if i2 in s1:
			return (w1,w2,s1,s2)
		#else, add nodes visited to the walks and s1,s2
		s1.add(i1)
		s2.add(i2)
		w1.append(i1)
		w2.append(i2)






def initialize_cuts(n,A,walker1,walker2,init_method,num_cut_init_walks=100):
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

def gen_votes(walker,S,S_comp,n,num_rounds = 1):
	'''In each iteration, run a sequence of walks so each vertex is seen
	at least once. For each vertex not in S or S_comp, we maintain
	a pair of votes that it should belong to S or S_comp.
	Votes correspond to the number of times a vertex is on walks that hit
	S or S_comp first.'''
	print('gen votes')
	votes = np.zeros((n,2))
	total_walks = [[]]*num_rounds
	#S_first, S_comp_first partitions nodes based on the first walk
	#it was on and which of S or S_comp the walk hit.
	S_first = copy.deepcopy(S)
	S_comp_first = copy.deepcopy(S_comp)
	partial_all = S.union(S_comp)
	num_in_partial_all = len(partial_all)
	placed_all_rounds = copy.deepcopy(partial_all)
	max_walks = 1000
	for round_num in range(num_rounds):
		start = time.time()
		walks_in_round = []
		'''Start walks from unseen nodes'''
		unseen_list = cut_walk_gen_utils.complement([S,S_comp],n)
		unseen = set(unseen_list)
		num_placed = num_in_partial_all
		walker.set_current(unseen_list)
		walk_gen = walker.walk_single
		#initialize walk
		w = [walker.current_node]
		s = set(w)
		#seen used to verify that nodes is either in S or S_comp, unseen
		#or seen
		seen = set()
		while num_placed < n and len(walks_in_round) < max_walks:
			'''Walk unti we hit the partial cut'''
			assert(len(partial_all) + len(seen) + len(unseen) == n)
			i = next(walk_gen())
			placed_all_rounds.add(i)
			'''Once we hit, for each vertex on the walk, 
			one vote for the side of the partial cut it hits first'''
			if i in S:
				for u in w:
					votes[u][0] = votes[u][0]+1
			if i in S_comp:
				assert(i not in S)
				for u in w:
					votes[u][1] = votes[u][1]+1
			'''Record the walk, start over outside the nodes we have seen'''
			if i in partial_all:
				#move nodes in s from rest to seen
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
				if num_placed < n:
					walker.set_current(list(unseen))
					walk_gen = walker.walk_single
					w = [walker.current_node]
					s = set(w)
			else:
				#add i to s and w
				s.add(i)
				w.append(i)
		assert((num_placed == n) or (len(walks_in_round) == max_walks))
		total_walks[round_num] = walks_in_round
	return votes, total_walks


def gen_cut_from_walks(A,walker1,walker2,walk_params,init_size_thresh = 5):
	'''Generate cut using random walks. Initialize the cut by generating
	two disjoint subsets of nodes S,S_comp. Remaining nodes will be merged with the subsets 
	by labeling the nodes with S,S_comp based on random walks that contain the node. Set S
	is the generated cut.
	'''
	init_method = walk_params['init_method']
	batch_size = walk_params['bs']
	num_cut_init_walks = walk_params['num_cut_init_walks']
	n = A.shape[0]
	S = set()
	S_comp = set()
	'''Cut initialization'''
	while min(len(S),len(S_comp)) < init_size_thresh:
		(w1,w2,S,S_comp) = initialize_cuts(n,A,walker1,walker2,init_method,num_cut_init_walks)
	walks = []
	nodes_to_label = cut_walk_gen_utils.complement([S,S_comp],n)
	assert((len(S)+len(S_comp)+len(nodes_to_label))==n)
	'''Cut construction. If nodes left to label, generate batch size number of walk algorithm 
	iterations and use these collections of walks to vote'''
	if len(nodes_to_label)>0:
		assert(len(S.intersection(S_comp))==0)
		votes, walks = gen_votes(walker1,S,S_comp,n,batch_size)
		for u in nodes_to_label:
			if votes[u][0] > votes[u][1]:
				S.add(u)
			else:
				S_comp.add(u)
	#making sure we found a cut
	assert(len(S) + len(S_comp) == n)
	assert(len(S.intersection(S_comp))==0)
	if len(nodes_to_label) == 0:
		walks = [[w1]+[w2]]
	else:
		walks = [[w1]+[w2]+walks[i] for i in range(batch_size)]
	return (S,S_comp,walks)

def num_nodes_in_cut(S,S_comp,s):
	'''Return the number of nodes before crossing cut'''
	if s[0] in S:
		walk_sign = 1
	else:
		walk_sign = -1
	walk_len = len(s)
	exited_cut = False
	for i in range(walk_len):
		s_i = s[i]
		if s_i in S and walk_sign == -1:
			exited_cut = True
			break
		elif s_i in S_comp and walk_sign == 1:
			exited_cut = True
			break
	if exited_cut:
		return i
	else:
		return walk_len

def comp_updates_node_dist(walk, distance, walk_dist_weight,weights, x, y):
	'''Compute pairs of nodes distance steps apart on walk. If walk_dist_weight,
	weight by distance. Append new weights and new vertices to pairs.'''
	start_nodes = walk[:-distance]
	end_nodes = walk[distance:]
	#symmetric, need updates in both direction
	weights_new = np.ones(len(start_nodes)*2)
	if walk_dist_weight:
		#weight inversely proportional to distance of nodes
		weights_new = weights_new*(1.0/float(distance))
	#updates = updates + sp.coo_matrix((weights,(start_nodes+end_nodes,end_nodes+start_nodes)),shape=(n,n)).toarray()
	weights = np.append(weights,weights_new)
	x = np.append(x,start_nodes)
	x = np.append(x,end_nodes)
	y = np.append(y,end_nodes)
	y = np.append(y,start_nodes)
	return weights,x,y

'''Using cuts and walk, update Frequency matrix using counts of node
pairs on the walks and the cut for additional information'''
def update_F(walks,F,walk_dist_weight,cut_disc,S,S_comp):
	#walks: list of walks generated, update for each walk in walks
	#F: frequency matrix to update
	#walk_dist_weight: If true, increase F(u,v) inversely proportional 
	#to the distance of (u,v) on walk 
	#cut_disc: If true, only increase F(u,v) if (u,v) appear before 
	#the walk crosses the cut
	#S,S_comp: the cut
	num_nodes = F.shape[0]
	assert(len(S)+len(S_comp)==num_nodes)
	updates = np.zeros_like(F)
	weights = np.array([])
	x = np.array([])
	y = np.array([])
	i = 0
	total_seen_on_walks = set()
	start = time.time()
	max_dist = 20
	for walk in walks:
		for vertex in walk:
			total_seen_on_walks.add(vertex)
		i = i+1
		if cut_disc:
			n = min(max_dist,num_nodes_in_cut(S,S_comp,walk))
		else:
			n = min(max_dist,len(walk))
		#update matrix for nodes in s dist apart
		for dist in range(1,n):
			weights, x, y = comp_updates_node_dist(walk, dist, walk_dist_weight,weights, x, y)
	print('time to compute matrix updates {}'.format(time.time()-start))
	updates = sp.coo_matrix((weights,(x,y)),shape=(num_nodes,num_nodes)).toarray()
	#no self loops
	np.fill_diagonal(updates,0)
	return F + updates, total_seen_on_walks

def matrix_update(A, S, S_comp, batch_size, F, walk_dist_weight, zero_cross, num_iters, walks, stats, round_num, truth_spec, frequencies):
	'''Given cut S,S_comp and set of walks used to find S,S_comp, update F 
	according to walk_dist_weight and zero_cross params.'''
	'''Return 
	updated F
	stats which maps statistic names to a list of statistic values on updated frequency matrices
	frequencies which is a map from num_iters to frequency matrices after num_iters'''
	total_seen = set()
	for j in range(batch_size):
		print('update for {}th set of walks generated in round {}'.format(j,round_num))
		start = time.time()
		'''Either computed batch size number of walks, or the initial walks
		labeled all nodes'''
		if (len(walks) == 1):
			F, seen = update_F(walks[0],F,walk_dist_weight,zero_cross,S,S_comp)
			total_seen = total_seen.union(seen)
			walk_num = 0
			final_walk = 0
			#use F for every walk algorithm iteartions in this batch
			for j in range(batch_size):
				if (round_num*batch_size)+j+1 in num_iters:
					#save the frequency matrix in num_iters
					normF = utils.normMatrix_wsum(F,A.sum())
					frequencies[(round_num*batch_size)+j+1] = normF
		elif (len(walks) > 1):
			assert(len(walks)==batch_size)
			F, seen = update_F(walks[j],F,walk_dist_weight,zero_cross,S,S_comp)
			total_seen = total_seen.union(seen)
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
		if (len(walks) > 1) and ((round_num*batch_size)+j+1 in num_iters):
			#save the frequency matrix in num_iters
			normF = utils.normMatrix_wsum(F,A.sum())
			frequencies[(round_num*batch_size)+j+1] = normF
		#Record statistics oncee all walk algorithim iterations recorded 
		#for every other round
		if (j==final_walk) and (round_num%2 == 1):
			if ((round_num*batch_size)+j) not in num_iters: 
				normF = utils.normMatrix_wsum(F,A.sum())
			spec_check = utils.spectrum(normF)
			l2_lin_check = utils.l2_lin_weight(spec_check,truth_spec)
			entropy_check = np.mean(utils.entropy_m(normF))
			stats['checkpoints'].append((round_num+1)*batch_size)
			stats['num_rounds'].append(round_num+1)
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
	print(len(total_seen))
	return F, stats, frequencies


def gen_freq_from_walks(A,walker1,walker2,num_iters,walk_params,plot_path=None,freq_path = None):
	'''Perform walk algorithm k times to construct frequency matrix'''
	'''Return map of num_iters to frequencies after num_iters'''
	start = time.time()
	times = {}
	n = A.shape[0]
	truth_spec = utils.spectrum(A)
	F = np.zeros((n,n))
	#stats on F as after completed rounds
	stats = defaultdict(list)
	#In each round, we run batch_size number of walk_alg iterations
	batch_size = walk_params['bs']
	walk_dist_weight = walk_params['walk_dist_weight']
	zero_cross = walk_params['zero_cross']
	num_rounds = [int(x/batch_size) for x in num_iters]
	frequency_map = {}
	#We output the graphs for each num_iter in num_iters_scaled. we run the
	#algorithm until the maximum is hit
	max_rounds = max(num_rounds)
	for i in range(max_rounds):
		print('Round {}'.format(i))
		#One cut generated per round with batch of sets of walks that label 
		#each node at least once
		print('time starting round {} {}'.format(i, time.time()-start))
		(S,S_comp,walks) = gen_cut_from_walks(A,walker1,walker2,walk_params)
		print('time after generating cuts in round {} {} '.format(i, time.time()-start))
		assert(len(S.intersection(S_comp)) == 0)
		F, stats, frequency_map = matrix_update(A, S, S_comp, batch_size, F, walk_dist_weight, zero_cross, num_iters, walks, stats, i, truth_spec, frequency_map)
		print('time after updating frequency matrix {} {} '.format(i, time.time()-start))
		#record time
		if ((i+1)*batch_size) in num_iters:
			if freq_path != None:
				print('SAVING HERE')
				sF = utils.normMatrix_wsum(F,A.sum())
				np.savetxt(freq_path+'num_iters{}.txt'.format((i+1)*batch_size),sF)
			times[(i+1)*batch_size] = time.time()-start
	assert(len(list(frequency_map.keys())) == len(num_iters))
	if plot_path !=None:
		with open(plot_path+"cuts.csv","w") as f:
		    wr = csv.writer(f)
		    wr.writerows(stats['cuts'])
		f.close()
		with open(plot_path+"first_walk_nodes.csv","w") as f:
		    wr = csv.writer(f)
		    wr.writerows(stats['first_walk_nodes'])
		f.close()
		np.savetxt(plot_path+'conductance.txt',stats['conductance'])
		np.savetxt(plot_path+'checkpoints.txt',stats['checkpoints'])
		np.savetxt(plot_path+'num_rounds.txt',stats['num_rounds'])
		np.savetxt(plot_path+'l2_lin.txt',stats['l2_lins'])
		np.savetxt(plot_path+'num_expect_edges.txt',stats['num_expect_edges'])
		np.savetxt(plot_path+'entropy.txt',stats['entropy_list'])
		np.savetxt(plot_path+'spectra.txt',stats['spectra'])
		np.savetxt(plot_path+'num_walks.txt',stats['num_walks'])
		np.savetxt(plot_path+'mean_walk_len.txt',stats['mean_walk_len'])
		np.savetxt(plot_path+'total_transitions.txt',stats['total_transitions'])
		np.savetxt(plot_path+'time_to_update.txt',stats['time_to_update'])
	return frequency_map, times



def rw_gen(A_truth,algs,num_iters,freq_paths, data_paths):
	'''Return frequency matrix and time for each alg in algs'''
	frequencies = {}
	times = {}
	walker1, walker2 = gen_walkers(A_truth)
	for alg in algs:
		print(alg)
		comp, config = gen_walk_config(alg)
		if comp == True:
			freq_path = freq_paths[alg]
			data_path = data_paths[alg]
			num_iters_alg = num_iters[alg]
			frequencies[alg], times[alg] = gen_freq_from_walks(A_truth,walker1,walker2,num_iters_alg,config,data_path,freq_path)
		else:
			print(alg_missing)
	return times





	
