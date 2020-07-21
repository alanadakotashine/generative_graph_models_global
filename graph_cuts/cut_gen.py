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


'''UTILS'''

def update_cut_results(F,c,c_comp,heavy,stats,results,A_truth):
	for stat in stats:
		if stat == 'heavy':
			results[stat].append(heavy)
		elif stat == 'diff':
			results[stat].append(cut_conn(F-A_truth,c,c_comp))
		elif stat == 'abs_diff':
			D = F-A_truth
			results[stat].append(cut_conn_abs(F-A_truth,c,c_comp))
		elif stat == 'cond_truth':
			results[stat].append(utils.compConductance(c,c_comp,A_truth))




'''CONFIGS FOR ALGORITHMS'''

def gen_cut_correct_config(cut_method, grasp_neighborhood_size):
	cut_params = {}
	cut_params['f']=vertex_conn
	cut_params['f_global']=cut_conn
	cut_params['frac_nodes_init'] = .2
	cut_params['num_grasp_repeats'] = 10
	cut_params['grasp_neighborhood_size'] = grasp_neighborhood_size
	if cut_method == 'goemmans-williams_independent_single_unifup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=True 
		cut_params['single']=True
		cut_params['unifup']=True
	if cut_method == 'goemmans-williams_independent_single_pushup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=True 
		cut_params['single']=True
		cut_params['unifup']=False
	if cut_method == 'goemmans-williams_independent_triple_unifup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=True 
		cut_params['single']=False
		cut_params['unifup']=True
	if cut_method == 'goemmans-williams_independent_triple_pushup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=True 
		cut_params['single']=False
		cut_params['unifup']=False
	if cut_method == 'goemmans-williams_recursive_single_unifup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=False
		cut_params['single']=True
		cut_params['unifup']=True
	if cut_method == 'goemmans-williams_recursive_single_pushup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=False
		cut_params['single']=True
		cut_params['unifup']=False
	if cut_method == 'goemmans-williams_recursive_triple_unifup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=False
		cut_params['single']=False
		cut_params['unifup']=True
	if cut_method == 'goemmans-williams_recursive_triple_pushup':
		cut_params['opt']='goemmans-williams'
		cut_params['independent']=False
		cut_params['single']=False
		cut_params['unifup']=False
	if cut_method == 'grasp_max_independent_single_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=True
		cut_params['unifup']=True
		cut_params['g']=identity
	if cut_method == 'grasp_max_independent_single_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=True
		cut_params['unifup']=False
		cut_params['g']=identity
	if cut_method == 'grasp_max_independent_triple_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=False
		cut_params['unifup']=True
		cut_params['g']=identity
	if cut_method == 'grasp_max_independent_triple_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=False
		cut_params['unifup']=False
		cut_params['g']=identity
	if cut_method == 'grasp_max_recursive_single_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=True
		cut_params['unifup']=True
		cut_params['g']=identity
	if cut_method == 'grasp_max_recursive_single_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=True
		cut_params['unifup']=False
		cut_params['g']=identity
	if cut_method == 'grasp_max_recursive_triple_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=False
		cut_params['unifup']=True
		cut_params['g']=identity
	if cut_method == 'grasp_max_recursive_triple_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=False
		cut_params['unifup']=False
		cut_params['g']=identity
	if cut_method == 'grasp_abs_independent_single_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=True
		cut_params['unifup']=True
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_independent_single_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=True
		cut_params['unifup']=False
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_independent_triple_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=False
		cut_params['unifup']=True
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_independent_triple_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=True 
		cut_params['single']=False
		cut_params['unifup']=False
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_recursive_single_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=True
		cut_params['unifup']=True
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_recursive_single_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=True
		cut_params['unifup']=False
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_recursive_triple_unifup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=False
		cut_params['unifup']=True
		cut_params['g']=np.abs
	if cut_method == 'grasp_abs_recursive_triple_pushup':
		cut_params['opt']='grasp'
		cut_params['independent']=False
		cut_params['single']=False
		cut_params['unifup']=False
		cut_params['g']=np.abs
	return cut_params






'''TESTS'''

def test_correct_balance_batch():
	#2 cuts
	#LL, LR, LC, RL, RR, RC, CL, CR, CC
	A_star = np.zeros((8,8))
	A_star[0,1]=.9
	A_star[0,3]=.1
	A_star[1,2]=1
	A_star[2,3]=1
	A_star[3,4]=1
	A_star[4,5]=1
	A_star[5,6]=1
	A_star[6,7]=1
	A_star[0,7]=1
	A_star = A_star + A_star.T 
	A = copy.deepcopy(A_star)
	A[4,5]=0
	A[5,4]=0
	A[1,5]=1
	A[5,1]=1
	cuts = [set([0,1,2,3])]
	for single in [True, False]:
		for unif_update in [True, False]:
			print('single {} unif update {}'.format(single, unif_update))
			A, high = correct_cut_set(A,A_star,cuts,single,unif_update)
			print(gen_connectivity_across_cuts(A,cuts))
			print(gen_connectivity_across_cuts(A_star,cuts))
	cuts = [set([0,1,2,3]),set([0,1,6,7])]
	for single in [True,False]:
		for unif_update in [True,False]:
			print('single {} unif update {}'.format(single, unif_update))
			A, high = correct_cut_set(A,A_star,cuts,single,unif_update)
			print(gen_connectivity_across_cuts(A,cuts))
			print(gen_connectivity_across_cuts(A_star,cuts))

def test_opt_assignment():
	#2 cuts
	#LL, LR, LC, RL, RR, RC, CL, CR, CC
	A_star = np.zeros((8,8))
	A_star[0,1]=1
	A_star[1,2]=1
	A_star[2,3]=1
	A_star[3,4]=1
	A_star[4,5]=1
	A_star[5,6]=1
	A_star[6,7]=1
	A_star[0,7]=1
	A_star = A_star + A_star.T 
	A = copy.deepcopy(A_star)
	A[4,5]=0
	A[5,4]=0
	A[1,5]=1
	A[5,1]=1
	cuts = [set([0,1,2,3])]
	single = True
	vertex_labels, node_part_label_vertices_map, variables, capacity_dict, capacities = construct_node_partition(A_star.shape[0],cuts)
	print(node_part_label_vertices_map)
	print(vertex_labels)
	print(variables)
	print(capacity_dict)
	#given a graph, we can compute the graph's assignment to each of the variables
	y_star = np.array(gen_assignment(A_star,node_part_label_vertices_map,variables))
	y_prime = np.array(gen_assignment(A,node_part_label_vertices_map,variables))
	#optimize for new assignment
	y = gen_opt_assignment(len(cuts), y_star, y_prime, capacities, single)
	print(y_star)
	print(y_prime)
	print(y)
	sum_differences = sum([(a-b)**2 for (a,b) in zip(y,y_prime)])
	I = np.eye(y_star.shape[0])
	print(sum_differences)
	print(np.matmul(np.matmul(y,I),y) - 2*np.matmul(y,y_prime) + np.matmul(np.matmul(y_prime,I),y_prime))
	alt_valid_solution = [4,2,2]
	print(sum([(a-b)**2 for (a,b) in zip(alt_valid_solution,y_prime)]))
	alt_valid_solution = [3,3,2]
	print(sum([(a-b)**2 for (a,b) in zip(alt_valid_solution,y_prime)]))
	alt_valid_solution = [2.5,3.5,2]
	print(sum([(a-b)**2 for (a,b) in zip(alt_valid_solution,y_prime)]))
	alt_valid_solution = [2.3,3.7,2]
	print(sum([(a-b)**2 for (a,b) in zip(alt_valid_solution,y_prime)]))

def gen_connectivity_across_cuts(A,cuts):
	result = []
	for c in cuts:
		cut_list = list(c)
		cut_list_comp = list(cut_walk_gen_utils.complement([cut_list],A.shape[0]))
		result.append(A[cut_list][:,cut_list_comp].sum())
	return result

	
def test_grasp():
	baseline = np.zeros((9,9))
	baseline[:3,:3] = [[0,1,1],[1,0,1],[1,1,0]]
	baseline[3:6,3:6]=[[0,1,1],[1,0,1],[1,1,0]]
	baseline[6:9,6:9]=[[0,1,1],[1,0,1],[1,1,0]]
	A = copy.deepcopy(baseline)
	A_truth = copy.deepcopy(baseline)
	print(A.shape)
	A_truth[0,3]=1
	A_truth[3,0]=1
	A_truth[1,4]=1
	A_truth[4,1]=1
	A_truth[2,5]=1
	A_truth[5,2]=1
	A_truth[0,6]=1
	A_truth[6,0]=1
	A_truth[4,5]=0
	A_truth[5,4]=0
	A_truth[2,1]=0
	A_truth[1,2]=0


	A[1,7]=.5
	A[7,1]=.5
	A[1,8]=.5
	A[8,1]=.5
	A[0,3]=.2
	A[3,0]=.2
	A[3,6]=.2
	A[6,3]=.2
	A[4,2]=.2
	A[2,4]=.2
	A[5,7]=.2
	A[7,5]=.2
	A[5,8]=.2
	A[8,5]=.2
	
	n = 9
	new_neighborhood_size = 5
	cur_neighborhood_size = n
	init_size = 2
	num_grasp_iters = 10
	#initialize out neighbors and out graph
	#ith row of out neighbors are the neighbors of i
	#ith row of out graph is the connectivity of the ith node to each of it's neighbors
	I = A_truth - A
	print(A_truth)
	print(A)
	print(A_truth.sum())
	print(A.sum())
	out_graph = I
	out_graph, in_graph, out_neighbors, in_neighbors = sample_neighborhood_subset(n, new_neighborhood_size,I,out_graph,[np.arange(n)]*n,cur_neighborhood_size)
	for i in range(10):
		c, c_comp = grasp(I,out_graph,init_size,num_grasp_iters,vertex_conn,cut_conn,np.abs,out_neighbors,in_graph,in_neighbors,new_neighborhood_size)
		print('value')
		print(cut_conn(I,c,c_comp))
		print(np.abs(cut_conn(I,c,c_comp)))
		print('cut')
		print(c)
		print(hi)
	I = A-A_truth
	out_graph = I
	out_graph, in_graph, out_neighbors, in_neighbors = sample_neighborhood_subset(n, new_neighborhood_size,I,out_graph,[np.arange(n)]*n,cur_neighborhood_size)
	for i in range(10):
		c, c_comp = grasp(I,out_graph,init_size,num_grasp_iters,vertex_conn,cut_conn,np.abs,out_neighbors,in_graph,in_neighbors,new_neighborhood_size)
		print('value')
		print(cut_conn(I,c,c_comp))
		print(np.abs(cut_conn(I,c,c_comp)))
		print('cut')
		print(c)

def test_gen_cuts():
	method = 'grasp_abs_independent_triple_pushup'
	config = gen_cut_correct_config(method,.5)
	print(config)
	A_truth = np.zeros((9,9))
	A_truth[:3,:3] = [[0,1,1],[1,0,1],[1,1,0]]
	A_truth[3:6,3:6]=[[0,1,1],[1,0,1],[1,1,0]]
	A_truth[6:9,6:9]=[[0,1,1],[1,0,1],[1,1,0]]
	A = copy.deepcopy(A_truth)
	print(A.shape)
	A[0,3]=1
	A[3,0]=1
	A[1,4]=1
	A[4,1]=1
	A[2,5]=1
	A[5,2]=1
	A[0,6]=1
	A[6,0]=1
	A[1,7]=-1
	A[7,1]=-1
	A[1,8]=-1
	A[8,1]=-1
	c, c_comp = gen_cuts(config, 1, A_truth - A)
	print(c)
	print(c_comp)
	print('value')
	print(cut_conn(A_truth - A,c,c_comp))
	print(cut_conn(A_truth,c,c_comp))
	print(cut_conn(A,c,c_comp))
	print(np.abs(cut_conn(A_truth-A,c,c_comp)))





'''CUT FIX GENERATON'''
'''Helpers for computing valid assignment for cut constraint '''


def gen_vertex_labels(n,cuts,node_partitions):
	'''Return map of node partition label to list of vertices with that label 

	Args:
		n - number of vertices
		cuts - list of list of lenth k. kth element is the kth cut
		node_partitions - list of k-length bit vectors with 
			the kth bit indicating if node in kth cut'''
	labels = {}
	for node_part_rep in node_partitions: #initialize list for representation 
		labels[node_part_rep] = []
	for i in range(n):
		rep_of_i = '' #compute bit vector of node i
		for cut in cuts:
			if i in cut:
				rep_of_i = rep_of_i + '1'
			else:
				rep_of_i = rep_of_i + '0'
		labels[rep_of_i].append(i) #append i to its list
	return labels

def bit_str_comp(x):
	'''Return complement of bit string x'''
	result = ''
	for i in x:
		if i == '1':
			result = result + '0'
		else:
			result = result+ '1'
	return result

def var_to_node_part_pair_reps(node_pair_part):
	'''Return list of node partition pairs correspoinding to node pair partition

	Args:
		node_pair_part - k length vector with ith entry equal to 0,1,2 
			for if node pair is outside, in, or crossing ith cut'''
	num_cuts = len(node_pair_part)
	node_part_1 = np.array(['0']*num_cuts)
	indices_crossing = []
	for k in range(num_cuts): #fill entries for pair in cut and crosing
		val = node_pair_part[k]
		if val == 1:
			node_part_1[k] = val
		if val == 2:
			indices_crossing.append(k)
	num_crossing = len(indices_crossing)
	if num_crossing == 0: #Pairs do not cross cuts, all pairs in the same node partition
		return [("".join(list(node_part_1)),"".join(list(node_part_1)))]
	else: #compue node partitions that cross cuts the pairs cross
		crossing_reps = [np.binary_repr(i,width=num_crossing) for i in range(2**(num_crossing-1))]
		crossing_reps_comp = [bit_str_comp(x) for x in crossing_reps]
		crossing_rep_pairs = zip(crossing_reps, crossing_reps_comp)
		result = []
		for (x,y) in crossing_rep_pairs:
			a = np.copy(node_part_1)
			b = np.copy(node_part_1)
			a[indices_crossing] = list(x)
			b[indices_crossing] = list(y)
			result.append(("".join(list(a)),"".join(list(b))))
		return result


def gen_assignment(A,node_part_to_vertices,node_pair_parts):
	'''
	Returns:
		result - list with ith entry the amount of mass assigned to node_pair_parts[i]
	Args:
		A - probabilistic adjacency matrix 
		node_part_to_vertices - map of node partition to list of nodes 
		node_pair_parts - list of node pair partitions 
	'''
	result = []
	for var in node_pair_parts:
		pairs = var_to_node_part_pair_reps(var) #list of pairs of node partitions in node pair partition
		num_edges = 0
		for (x,y) in pairs:
			sub_graph = A[node_part_to_vertices[x]][:,node_part_to_vertices[y]]
			if x==y:
				num_edges += sub_graph.sum()/2
			else:
				num_edges += sub_graph.sum()
		result.append(num_edges)
	return result


def gen_capacities(node_pair_parts,vertex_labels):
	'''Compute number of node pairs (capacity ) in each node pair partition

	Return:
		capacity_map - map of each node pair partition to capacity
		capacity_list - list of capacities with ith entry equal to 
			capacity of ith node pair partition  

	Args:
		node_pair_parts - list of node pair partitions represented
			as vectors with each entry as 0,1,2 for if node pairs 
			are outside, inside, or crossing cut
		vertex_labels - map of each node partititon to list of nodes
	'''
	capacity_map = {}
	capacity_list = []
	for var in node_pair_parts:
		node_partition_pairs = var_to_node_part_pair_reps(var) #list of pairs of node partitions in node pair partition
		capacity = 0
		for (x,y) in node_partition_pairs: 
			if x == y:
				v = len(vertex_labels[x])
				capacity += (v*(v-1))/2 #do not count each edge twice
			else:
				capacity += len(vertex_labels[x])*len(vertex_labels[y])
		capacity_map[var] = capacity
		capacity_list.append(capacity)
	return capacity_map, capacity_list


def gen_constraints(num_cuts, variables):
	'''Compute constraint matrix for 
		number of edges on either side and crossing num_cuts cuts 

	Return:
		constraints - num_cuts*3 by num_variables matrix with 
			the ith triple of rows denoting the three constraints 
			for the ith cut 

	Args:
		num_cuts - Int for number of cuts
		variables - Representation for each node pair partition with 
			ith entry taking 0,1,2 marking if node pairs 
			are outside, inside, or crossing the ith cut

	'''
	num_variables = len(variables)
	constraints = np.zeros((num_cuts*3,num_variables))
	for i in range(num_variables):
		var = variables[i]
		for cut_num in range(num_cuts):
			constraints[(cut_num*3)+var[cut_num]][i] = 1
	return constraints

def gen_constraints_cross_only(num_cuts, variables):

	'''Compute constraint matrix for 
		number of edges crossing num_cuts cuts

	Return:
		constraints - num_cuts*3 by num_variables matrix with 
			the ith triple of rows denoting the three constraints 
			for the ith cut 

	Args:
		num_cuts - Int for number of cuts
		variables - Representation for each node pair partition with 
			ith entry taking 0,1,2 marking if node pairs 
			are outside, inside, or crossing the ith cut

	'''
	num_variables = 3**num_cuts
	constraints = np.zeros((num_cuts,num_variables))
	for i in range(num_variables):
		var = variables[i]
		for j in range(num_cuts):
			if var[j] == 2: #Node pair is crossing jth cut
				constraints[j][i] = 1
	return constraints

'''Helpers for updating subgraphs'''

def update_helper(sub_graph, diff,unif_update=False):
	'''Change the sum of the entries in the sub_graph by uniformly on 
	pairs with space or uniformly on pairs with space above/below midpoint

	Return: 
		sub_graph - matrix with diff amount of mass added with entries within [0,1] 

	Args:
		sub_graph - matrix with entries within [0,1]
		diff - amount to add, could be positive or negative 
		unif_update - Boolean flag for method to distribute diff 
			among pairs in sub_graph (default is False)
			If True, use unif_update. Use push_update otherwise. '''
	t_first = .001
	total_before = sub_graph.sum()
	left = diff
	#if left is positive, we are adding. If negative, removing
	t = t_first
	#Add/remove value if entry is above/below median in
	#order to add/remove to entries already high/low
	mid_value = np.median(sub_graph[sub_graph>t])
	while (np.abs(left) > t_first):
		#removing
		if left < 0:
			#compute satisfying pairs, must make progress (have at least t to remove)
			#and if not unif_update, only remove from values belwo mid value
			if not unif_update:
				num_above = len(sub_graph[(sub_graph>t) & (sub_graph <= mid_value)])
			else:
				num_above = len(sub_graph[(sub_graph>t)])
			#if satisfyings set empty, relax constraints
			if num_above == 0:
				if unif_update:
					t = t/2.0
				else:
					mid_value = min(1.0,mid_value+.1)
					if mid_value == 1.0:
						t = t/2.0
				continue
			if not unif_update:
				#max i can remove
				bottleneck = np.min(sub_graph[(sub_graph>t) & (sub_graph <= mid_value)]) 
				#only remove amout i need to
				delta = min(bottleneck,(-1*left)/num_above)
				#removal
				sub_graph[(sub_graph>t) & (sub_graph <= mid_value)] = sub_graph[(sub_graph>t) & (sub_graph <= mid_value)] - delta
			else:
				bottleneck = np.min(sub_graph[(sub_graph>t)]) 
				delta = min(bottleneck,(-1*left)/num_above)
				sub_graph[(sub_graph>t)] = sub_graph[(sub_graph>t)] - delta
			#recompute how much i have left
			left += delta*num_above
		#adding
		else:
			#only look at adding values that have at least t amount of room
			if not unif_update:
				num_above = len(sub_graph[(sub_graph<(1-t)) & (sub_graph >= mid_value)])
			else:
				num_above = len(sub_graph[(sub_graph<(1-t))])
			if num_above == 0:
				#relax constarints
				if unif_update:
					t = t/2.0
				else:
					mid_value = max(0.0,mid_value-.1)
					if mid_value == 0.0:
						t = t/2.0
				continue
			if not unif_update:
				bottleneck = 1 - np.max(sub_graph[(sub_graph<(1-t)) & (sub_graph >= mid_value)]) 
				delta = min(bottleneck, left/num_above)
				sub_graph[(sub_graph<(1-t)) & (sub_graph >= mid_value)] = sub_graph[(sub_graph<(1-t)) & (sub_graph >= mid_value)] + delta
			else:
				bottleneck = 1 - np.max(sub_graph[(sub_graph<(1-t))]) 
				delta = min(bottleneck, left/num_above)
				sub_graph[(sub_graph<(1-t))] = sub_graph[(sub_graph<(1-t))] + delta
			left -= delta*num_above
	assert(np.abs((total_before + diff) - sub_graph.sum()) <= t_first)
	return sub_graph

def remove_add_space_norm(pairs,A,vertex_labels,delta):
	'''Map the node partition pairs to the amount the space they have left
	to add mass, the amount of mass they have to remove, the number of pairs
	and the matrix itself defined by the pair of nodes.'''
	add_space = {}
	remove_space = {}
	sub_graph_size = {}
	sub_graphs = {}
	for (x,y) in pairs:
		v1 = vertex_labels[x]
		v2 = vertex_labels[y]
		sub_graph = A[v1][:,v2]
		if x == y:
			sub_graph_size[(x,y)] = max(1.0,float((len(v1)*(len(v1)-1))/2))
		else:
			sub_graph_size[(x,y)] = max(1.0,float(len(v1)*len(v2)))
		sub_graphs[(x,y)] = sub_graph
		#prior_sums.append(sub_graph.sum())
		#self loops
		remove_space[(x,y)] = sub_graph.sum()
		if v1 == v2:
			if delta > 0:
				#Fill diagonal with 1 so there is no space to add, will remove after we 
				#update subgraph
				np.fill_diagonal(sub_graph,1)
		add_space[(x,y)] = (1-sub_graph).sum()
	return add_space, remove_space, sub_graph_size, sub_graphs

def comp_subgraph_values(add_space, remove_space, sub_graph_size, delta, pairs):
	'''Weight each pair of node partitions inversely proportional to the amount of space
	we have to add/remove to give heavy/light pairs more to add/more to remove. '''
	#pair_weights is a list of tuples from the pairs of vertices to the amount of 
	#space the subgrpah has to add/remove mass normalized by the size of the subgraph
	if delta > 0:
		#adding, want to add to values that have most of the edges filled.
		pair_weights = [((x,y),remove_space[(x,y)]/sub_graph_size[(x,y)]) for (x,y) in pairs]
	else:
		#removing, want to remove from values that have most of the edges empty.
		pair_weights = [((x,y),add_space[(x,y)]/sub_graph_size[(x,y)]) for (x,y) in pairs]
	total_value = sum([x[1] for x in pair_weights])
	#if space is zero on all pairs, assign same value to all
	if total_value == 0:
		new_pair_weights = {}
		for pair in pairs:
			new_pair_weights[pair] = 1
		pair_weights = new_pair_weights.items()
		total_value = sum([x[1] for x in pair_weights])
	#use the weight to assign the value to add/remove to each set of pairs
	sorted_pair_weights = sorted(pair_weights, key = lambda x: x[1], reverse=True)
	pair_values = {}
	for ((x,y),v) in sorted_pair_weights:
		pair_values[(x,y)] = (delta * (v/total_value))
	return pair_values, sorted_pair_weights



def comp_pair_weight(pairs,A,delta,vertex_labels):
	'''Assign pair of node sets mass to add/remove proportional to their value conditoned
	on not adding more than there is space or removing more than there is already mass 
	assgigned.'''
	add_space, remove_space, sub_graph_size, sub_graphs = remove_add_space_norm(pairs,A,vertex_labels, delta)
	pair_values, sorted_pair_weights = comp_subgraph_values(add_space, remove_space, sub_graph_size, delta, pairs)
	surplus = 0
	#barometer to complete
	bar = .00000001
	while True:
		#visit in order to try to use up all the weight
		for ((x,y),v) in sorted_pair_weights:
			assigned = pair_values[(x,y)]
			#attempt to use up the surplus that we couldn't use before
			target = assigned+surplus
			if delta > 0:
				space = add_space[(x,y)]
				actual = min(target, space)
				surplus = surplus+assigned-actual
				pair_values[(x,y)] = actual
			else:
				space = remove_space[(x,y)]
				actual = min(space, -1*target)
				surplus = surplus+assigned+actual
				pair_values[(x,y)] = -1*actual
		if np.abs(surplus) < bar:
			break
	assert(np.abs(sum(list(pair_values.values())) - delta) < bar)
	return pair_values, sub_graphs


def sub_graph_assignment(A,diffs,current,new_sol,variable_vertex_map,variables,unif_update=False):
	'''Diffs tells us how much each of the variabels change from the current solution
	to the new solution'''
	'''variables: list of the node pair partition representation'''
	'''vertex_labels: map of node partition representations to vertices in those partitions'''
	'''Returns pair_weight_all: Map of pairs of node partitions to how much to add/remove 
	from subgraph defined by pair of node partitions'''
	'''Returns sub_graphs_all: Map of pairs of node partitions to the adjacency matrix of the subgraph
	defined by the pair of node partitions'''
	capacities, _ = gen_capacities(variables,variable_vertex_map)
	pair_weight_all = {}
	sub_graphs_all = {}
	for var in diffs:
		diff = diffs[var]
		capacity = capacities[var]
		cur = current[var]
		new_val = new_sol[var]
		#adjust diff to ensure we aren't removing more than there is or adding beyond capacity
		if diff < 0:
			diff_actual = max(-1*(cur),diff)
		if diff > 0:
			diff_actual = min(capacity - cur, diff)
		if (np.abs(diff) > 0):
			if np.abs(diff_actual) > 0:
				pairs = var_to_node_part_pair_reps(var) #list of pairs of node partitions in node pair partition
				#how much to add/remove from each sub-graph depends
				#on how much space it has
				pair_weight, sub_graphs = comp_pair_weight(pairs,A,diff_actual,variable_vertex_map)
				pair_weight_all.update(pair_weight)
				sub_graphs_all.update(sub_graphs)
	return pair_weight_all, sub_graphs_all

def update_sub_graphs(A, pair_weight_all, sub_graphs_all, vertex_labels, unif_update=False):
	'''This updates A to the new solution by updating the subgraphs each of the variabels maps to
	by diff value'''
	pairs = list(pair_weight_all.keys())
	num_pairs = len(pairs)
	for i in range(num_pairs):
		(x,y) = pairs[i]
		v1 = vertex_labels[x]
		v2 = vertex_labels[y]
		delta = pair_weight_all[(x,y)]
		if x == y:
			delta = delta*2
		before = sub_graphs_all[(x,y)].sum()
		sub_graph = update_helper(sub_graphs_all[(x,y)],delta,unif_update)
		if v1 == v2:
			 np.fill_diagonal(sub_graph,0)
			 A[np.ix_(v1,v2)] = sub_graph
		else:
			A[np.ix_(v1,v2)] = sub_graph
			A[np.ix_(v2,v1)] = sub_graph.T
	return A

'''cut sampling helpers'''	
def random_place(n,i):
	#print('random choose {} from {}'.format(i,n))
	assert(i<=n)
	#Randomly permute nodes
	rand_order = np.random.permutation(range(n))
	S = set([rand_order[0]])
	S_comp = set([rand_order[1]])
	#flip coin for first i nodes in the sequence to place
	#in S or S_comp
	for j in range(2,i):
		p = np.random.binomial(1,.5)
		if p == 1:
			S.add(rand_order[j])
		else:
			S_comp.add(rand_order[j])
	assert(len(S) + len(S_comp) == i)
	return (S,S_comp)

def neg_connectivity_abs(A,S,u):
	return -1*abs(connectivity(A,S,u))

def neg_connectivity(A,S,u):
	return -1*connectivity(A,S,u)

def connectivity(A,S,u):
	assert(u not in S)
	conn_edges = A[np.ix_(S,[u])]
	return conn_edges.sum()

def comp_incoming_neighbors(outgoing,n,A):
	'''outgoing is a list of lists. vth list is list of vs outgoing neighbors'''
	'''Returns: incoming is a list of lists. vth list is list of vs incoming neighbors'''
	'''incoming_neighbors_conn is list of lists containing connectivity of v to it's neighbors'''
	incoming_neighbors = []
	incoming_neighbors_conn = []
	for i in range(n):
		incoming_neighbors.append([])
		incoming_neighbors_conn.append([])
	#For each edge from  to v, add u to v's incoming neighbors
	for i in range(n):
		neighbors = outgoing[i]
		for j in neighbors:
			incoming_neighbors[j].append(i)
			incoming_neighbors_conn[j].append(A[i,j])
	return incoming_neighbors, incoming_neighbors_conn

def sample_neighborhood_subset_helper(A,A_sub,new_num_neighbors,cur_num_neighbors,cur_neighbors,n):
	'''Sample new_num_neighbors from cur_neighhbors of for each node'''
	'''neighbors is a list of neighbor lists'''
	'''outgoing is the connectivity for each node to it's neighbors'''
	assert(A_sub.shape[1] == cur_num_neighbors)
	#Can't subsample
	if new_num_neighbors >= cur_num_neighbors:
		return cur_neighbors, A_sub
	else:
		#add noise in case connectivity of node is zero
		min_entry = max(np.min(np.abs(A_sub)),.0000000001)
		abs_A_noise = np.abs(A_sub) + (min_entry/100.0)
		Z = np.sum(abs_A_noise,axis=1).astype(float)
		P = abs_A_noise/Z[:,None]
		#initialize
		neighbors = np.zeros((n,new_num_neighbors)).astype(int)
		outgoing = np.zeros((n,new_num_neighbors))
		for i in range(n):
			#indices of the new neighbors in the cur neighbor lists
			sub_sample = np.random.choice(cur_num_neighbors,new_num_neighbors,p=P[i],replace=False).astype(int)
			#new neighbor lists
			neighbors[i] = cur_neighbors[i][sub_sample]
			#connectivity to new neighbors
			outgoing[i] = A[i][neighbors[i]]
	return neighbors, outgoing

def sample_neighborhood_subset(n, sub_neighborhood_size, A, out_graph, out_neighbors, neighborhood_size):
	'''n is the number of nodes'''
	'''sub_neighborhood_size is the resulting number of outgoing neighbors for each node'''
	'''A is the input graph'''
	'''out_graph[u][v] is the connectivity of the uth node to it's vth outgoing neighbor '''
	'''out_neighbors is a n length list of lists. the uth list contains the outgoing neighbors of u'''
	'''neighborhood_size is the current number of outgoing neighbors for each node'''
	assert(A.shape[0] == n)
	assert(A.shape[1] == n)
	assert(out_graph.shape[1]==neighborhood_size)
	assert(sub_neighborhood_size <= neighborhood_size)
	out_neighbors, out_graph = sample_neighborhood_subset_helper(A,out_graph,sub_neighborhood_size,neighborhood_size,out_neighbors,n)
	assert(out_graph.shape[1] == sub_neighborhood_size)
	'''Incoming neighbors of v are all of the nodes that contain v as a neighbor'''
	'''in_graph[u][v] is the connectivity of u to it's vth incoming neighbor '''
	in_neighbors, in_graph = comp_incoming_neighbors(out_neighbors,n,A)
	u = 5
	v = out_neighbors[u][2]
	return out_graph, in_graph, out_neighbors, in_neighbors


def conn_list_init_help(A,outgoing_neighbors,S,S_comp, n, conn_list):
	'''For each node i, compute its connectiivty to S,S_comp using only its
	outgoing neighbors'''
	#A is a n by n' matrix where n' is the number of outgoing neighbors
	#outgoing_neighbors is a n by n' matrix listing the neighbors of each node
	#S, S_comp lists the nodes in S and S_comp
	#conn_list[:][i] is the connecitivty of i to S and S_comp
	for i in range(n):
		neighbors_i = outgoing_neighbors[i]
		num_neighbors = len(neighbors_i)
		for j in range(num_neighbors):
			if neighbors_i[j] in S:
				conn_list[0][i] += A[i,j]
			elif neighbors_i[j] in S_comp:
				conn_list[1][i] += A[i,j]
	return conn_list

def conn_list_init(S,S_comp,n,A,out_graph,f,out_neighbors,in_graph, in_neighbors, neighborhood_size,sub_neighborhood_size):
	'''Store connectivity of each node to S and S_comp using only nodes it neighborhood'''
	conn_list = np.zeros((2,n))
	'''If sub neighborhood size is at least current neighborhood size, no need to sub sample'''
	if sub_neighborhood_size >= neighborhood_size:
		#If sub neighbrhood size is n, use entire graph
		if sub_neighborhood_size == n:
			conn_list[0] = A[list(S),:].sum(axis=0)
			conn_list[1] = A[list(S_comp),:].sum(axis=0)
		#Compute which of neighbors of i are in S and which are in s_comp
		else:
			conn_list = conn_list_init_help(out_graph,out_neighbors,S,S_comp, n, conn_list)
		return conn_list, in_neighbors, out_neighbors, in_graph, out_graph
	'''Else, sample sub_neighborhood_size neighbors from neighbors for each node'''
	out_graph, in_graph, out_neighbors, in_neighbors = sample_neighborhood_subset(n, sub_neighborhood_size,A,out_graph,out_neighbors,neighborhood_size)
	conn_list = conn_list_init_help(out_graph,out_neighbors,S,S_comp, n, conn_list)
	return conn_list, in_neighbors, out_neighbors, in_graph, out_graph


def cut_conn_abs(A,S,S_comp):
	return np.abs(cut_conn(A,S,S_comp))

def conversion(S,S_comp):
	return list(S), list(S_comp)

def extract(A,S,S_comp):
	return A[S][:,S_comp]

def cut_conn(A,S,S_comp):
	S_l, S_comp_l = conversion(S,S_comp)
	sub_mat = extract(A,S_l,S_comp_l)
	return sub_mat.sum()

def vertex_conn(A,u,S_comp):
	return A[u][:,list(S_comp)].sum()

def neg_cut_conn(A,S,S_comp):
	assert(len(S) + len(S_comp) == A.shape[0])
	return -1*(cut_conn(A,S,S_comp))

def comp_candidate_local_improvement(n,g,conn_list,cur_score,bitS,bitS_comp):
	move_to_S_temp = cur_score + conn_list[1]-conn_list[0]
	move_to_S = np.multiply(move_to_S_temp,bitS_comp)
	move_from_S_temp = cur_score + conn_list[0]-conn_list[1]
	move_from_S = np.multiply(move_from_S_temp, bitS)
	new_scores = g(np.add(move_to_S,move_from_S))
	return np.where(new_scores>g(cur_score))[0]


def permute_cands(candidates):
	num_candidates = len(candidates)
	rand_order = np.random.permutation(len(candidates))
	return num_candidates, rand_order


def local_cut_improvement_helper(candidates,conn_list,incoming,g,S,S_comp,cur_score,bitS,bitS_comp,incoming_neighbors):
	change = False
	num_candidates, rand_order = permute_cands(candidates)
	for i in range(num_candidates):
		v = candidates[rand_order[i]]
		#check to make sure is still candidate
		conn_v = conn_list[:,v]
		#swap u if still a candidate
		#print('checking if we should swap')
		conn_S = conn_v[0]
		conn_S_comp = conn_v[1]
		if (v in S) and (g(cur_score + conn_S - conn_S_comp) > g(cur_score)):
			change = True
			S.remove(v)
			S_comp.add(v)
			bitS[v]=0
			bitS_comp[v]=1
			cur_score = cur_score + conn_S - conn_S_comp
		elif (v in S_comp) and (g(cur_score + conn_S_comp - conn_S) > g(cur_score)):
			change = True
			S_comp.remove(v)
			S.add(v)
			bitS[v]=1
			bitS_comp[v]=0
			cur_score = cur_score + conn_S_comp - conn_S
		#update conn_list
		#placement effects all the neighbors of v
		if change == True:
			conn_list = update_conn_list(conn_list, incoming, v, S, incoming_neighbors)
	return change, conn_list, S, S_comp, cur_score, bitS,bitS_comp




def local_cut_improvement(A,S,S_comp,n,incoming,f,g,f_global,conn_list,bitS,bitS_comp,neighbors, cycle_length = 10):
	'''S and S_comp are sets'''
	'''conn_list maps every vertex to f(v,S) and f(v,S_comp)'''
	'''Make local improvements to S,S_comp by moving vertices such that 
	f_global(A,S,S_comp) incraeses'''
	cur_score = f_global(A,S,S_comp)
	candidates = comp_candidate_local_improvement(n,g,conn_list,cur_score,bitS,bitS_comp)
	#maintain previous candidate list to detect cycles
	candidates_prev = [candidates]
	change = True
	k = 0
	max_changes = 100
	while (change and (k < max_changes)):
		change = False
		#loop through candidates to find local improvements. Must be able to compute the effect
		#of switching v from g(f(v,S)) and g(f(v,S_comp))
		change, conn_list, S, S_comp, cur_score,bitS,bitS_comp = local_cut_improvement_helper(candidates,conn_list,incoming,g,S,S_comp,cur_score,bitS,bitS_comp,neighbors)
		k = k+1
		if change == True:
			candidates = comp_candidate_local_improvement(n,g,conn_list,cur_score,bitS,bitS_comp)
			#if candidates have already been seen, break
			if np.any([np.array_equal(i,candidates) for i in candidates_prev]):
				break
		#if the length of candidates is longer than the cycle length, no need to 
		#keep track of the first candidate in candidates_prev
		if len(candidates_prev) == cycle_length:
			candidates_prev.pop(0)
		#save current set of candidates
		candidates_prev.append(candidates)
		#if we moved at least one vertex as measured by chage, repeat
	return (S,S_comp)

def assignment_conn_list(inc,dec,incIndex,decIndex,conn_list,incoming,v,incoming_neighbors):
	'''inc/dec indicates if we moved v to/from'''
	'''incIndex/decIndex indicates where we moved v to/from'''
	'''to update conn_list, all the inverse neighbors of v will be affected'''
	'''For each of these neighbors, their connectivity to S/S_comp will be changed
	by how connected the node is to v'''
	neighbors_of_v = incoming_neighbors[v]
	if inc == True:
		conn_list[incIndex][neighbors_of_v] += incoming[v]
	if dec == True:
		conn_list[decIndex][neighbors_of_v] -= incoming[v]
	return conn_list


def update_conn_list(conn_list, incoming, v, S, incoming_neighbors, inc=True, dec = True):
	'''status of v has changed to be in/out of S, 
	increase/decrease the connectivity
	of nodes to S using the adjacency list of v'''
	if v in S:
		incIndex = 0
		decIndex = 1
	else:
		incIndex = 1
		decIndex = 0
	return assignment_conn_list(inc,dec,incIndex,decIndex,conn_list,incoming,v, incoming_neighbors)

def greedy_place(A,out_graph,s,f,g,out_neighbors,in_graph,in_neighbors,n,neighborhood_size,sub_neighborhood_size):
	'''Randomly place s nodes into S,S_comp disjoint sets'''
	(S,S_comp) = random_place(n,s)
	print(S)
	print(S_comp)
	#conn_list constists of two lists of length n
	#where the vth entry maps f(v,S(N(v))) where  S(N(v)) are the elements of 
	#of S that intersect N(v) which is the set of the neighbors of v
	conn_list, in_neighbors, out_neighbors, in_graph, out_graph = conn_list_init(S,S_comp,n,A,out_graph,f,out_neighbors,in_graph,in_neighbors,neighborhood_size,sub_neighborhood_size)
	'''Greedily place v with S/S_comp that maximizes g'''
	rand_order = np.random.permutation(n)
	bitS = np.zeros(n)
	bitS_comp = np.zeros(n)
	for v in rand_order:
		if v in S:
			bitS[v]=1
		elif v in S_comp:
			bitS_comp[v]=1
		else:
			conn_v = conn_list[:,v]
			print(conn_v)
			#if g(f(v,S)) is high, place on other side
			#to have high value of g(f(-)) across cut
			if g(conn_v[0])>=g(conn_v[1]):
				S_comp.add(v)
				bitS_comp[v]=1
			else:
				S.add(v) 
				bitS[v]=1
			print(S)
			#adding v for the first time, connectivity to either S,S_comp
			#is only increased for node neighbors so no need to decrement
			increment_update = True
			decrement_update = False
			#need all of v's incoming neighbors to compute the update to the connectivity list
			conn_list = update_conn_list(conn_list, in_graph, v, S, in_neighbors, increment_update, decrement_update)
	assert(len(S) + len(S_comp)==n)
	return (S,S_comp,conn_list,bitS,bitS_comp,in_neighbors,in_graph)






def grasp(A,out_graph,s,k,f,f_global,g,out_neighbors,in_graph,in_neighbors,sub_neighborhood_size):
	#A is entire adjacency matrix, ith row is connectivity of node i to all others
	#ith row of out_graph is connectivity of node i to it's neighbors in out_neighbors
	n = out_graph.shape[0]
	neighborhood_size = out_graph.shape[1]
	#A_sub only contains connectivity of v to their neighbors
	for i in range(k):
		random.seed(datetime.now())
		#Place s nodes randomly into disjoint sets
		#S,S_comp. Place ramining n-s 
		#nodes greedily by maximizing g(f(v,Q)) over Q in
		#taking S or S_comp. S,S_comp
		#at the end consist of all n nodes.
		(S,S_comp,conn_list,bitS,bitS_comp,in_neighbors,in_graph) = greedy_place(A,out_graph,s,f,g,out_neighbors,in_graph,in_neighbors,n,neighborhood_size,sub_neighborhood_size)
		init_score = score = g(f_global(A,S,S_comp))
		#make local improvements by swapping nodes v by maximizing f(S,S_comp)
		(S,S_comp) = local_cut_improvement(A,S,S_comp,n,in_graph,f,g,f_global,conn_list,bitS,bitS_comp,in_neighbors)
		score = g(f_global(A,S,S_comp))
		if i == 0:
			S_star = S
			S_comp_star = S_comp 
			cur_best = score
		elif score > cur_best:
			S_star = S
			S_comp_star = S_comp 
			cur_best = score
	return (S_star, S_comp_star)

def correct_cut_set_gen_constraints(P, capacities, true_conn, total):
	'''Computes constraint matrix with relaxations for connectivity and capcaity

	Returns:
		C - constraint matrix 
		b - value matrix 

	Args:
		P - connectivity matrix with ith row or ith triple of rows
			denoting connectivity constraint for ith cut (single or triple) 
		capacities - number of node pairs in each node pair partition defined by cuts
		true_conn - true connectivity across each cut
		total - total number of mass in true adjacency matrix '''
	relax_conn_constraint = .01
	relax_capacity_constraint = .01
	relax_total_constraint = .01
	num_cut_constraints = P.shape[0]*2
	num_vars = P.shape[1]
	C = np.zeros((num_cut_constraints + num_vars*2 + 2,num_vars))
	b = np.zeros(num_cut_constraints+num_vars*2 + 2)
	#conn constraints
	C[:int(num_cut_constraints/2)]=P
	C[int(num_cut_constraints/2):num_cut_constraints]=-1*P
	b[:int(num_cut_constraints/2)] = true_conn-relax_conn_constraint
	b[int(num_cut_constraints/2):num_cut_constraints] = -true_conn-relax_conn_constraint
	#non-negativty
	C[num_cut_constraints:num_cut_constraints+num_vars]=np.eye(num_vars)
	b[num_cut_constraints:num_cut_constraints+num_vars] = np.ones(num_vars)*-relax_capacity_constraint
	#capacity
	C[num_cut_constraints+num_vars:num_cut_constraints+2*num_vars] = -1*np.eye(num_vars)
	b[num_cut_constraints+num_vars:num_cut_constraints+2*num_vars]=(-1*np.array(capacities))-relax_capacity_constraint
	#total constraints
	C[num_cut_constraints+2*num_vars]=np.ones(num_vars)
	C[num_cut_constraints+2*num_vars+1]=-1*np.ones(num_vars)	
	b[num_cut_constraints+2*num_vars]=total-relax_total_constraint
	b[num_cut_constraints+2*num_vars+1]=-total-relax_total_constraint
	return C, b

def comp_heavy(true_conn, cur_conn, single, num_cuts):
	'''Compute for each cut if connectivity is across is heavier than truth 

	Returns: 
		heavy - vector of legth num_cuts that has 1 if connectivity across cut 
			is heavier than truth and 0 otherwise  

	Args:
		true_conn - true connectivity vector 
		cur_conn - current connectivity vector 
		single - Boolean flag, True if connectivity vectors contain connectivity 
			across only
		num_cuts - Int, number of cuts '''
	heavy = [0]*num_cuts 
	for i in range(num_cuts):
		if single:
			if cur_conn[i] > true_conn[i]:
				heavy[i] = 1
			elif  cur_conn[i] < true_conn[i]:
				heavy[i] = -1
		else:
			j = i*3 + 2 #connectivity across cut is last value in each triple
			if cur_conn[j] > true_conn[j]:
				heavy[i] = 1
			elif  cur_conn[j] < true_conn[j]:
				heavy[i] = -1
	return heavy

def gen_opt_assignment(num_cuts, y_star, y_prime, capacities, single, variables):
	'''Compute assignment of number of edges to place y 
	on each node pair set in variables that satisfies the cut constraints 
	imposed by true assignment y_star closest to current assignment y_prime

	Returns: 
		y - vector of number of edges to place in each node pair set in variables 
		heavy - indicator vector for whether current assignment across each cut 
			is heavier than true assignment

	Args:
		num_cuts - int number of cuts
		y_star - true assignment 
		y_prime - current assignment 
		variables - '''
	total = y_star.sum()
	solve_qp = True
	if single:
		P = gen_constraints_cross_only(num_cuts, variables) #one constraint for each cut
	else:
		P = gen_constraints(num_cuts, variables) #three constarints for each cut
		if num_cuts == 1: #only one solution
			solve_qp = False
	true_conn = np.matmul(P,y_star) 
	cur_conn = np.matmul(P,y_prime)
	heavy = comp_heavy(true_conn, cur_conn, single, num_cuts) #Compute if cuts are currently heavy/light
	if solve_qp:
		C,b = correct_cut_set_gen_constraints(P, capacities, true_conn, total) #add capacity constraints and total constraints
		I = np.eye(y_star.shape[0]) #minimized squared difference to y_prime
		y = quadprog.solve_qp(I,y_prime,C.T,b)[0]
	else:
		y = y_star
	return y, heavy

def construct_node_partition(n,cuts):
	'''Maps the partition of the graph defined by k cuts to the vertices

	Returns:
		vertex_labels - list of k-length bit vectors for each vertex
			representing membership of each vertex to the k cuts 
		node_part_label_vertices_map - map of k-length bit vector label 
			to vertices with that membership label 
		variables - representation for each node pair partition defined by cuts 
		capacity_dict - map of variables to number of node pairs 
		capcities - list where ith entry denotes the number of node pairs
			corrrespoinding to the ith variable 

	Args:
		n - number of vertices
		cuts - list of k lists of vertices 

	'''
	k = len(cuts)
	vertex_labels = [np.binary_repr(i,width=k) for i in range(2**k)]
	node_part_label_vertices_map = gen_vertex_labels(n,cuts,vertex_labels)
	variables = list(itertools.product([0,1,2], repeat = k))
	capacity_dict, capacities = gen_capacities(variables, node_part_label_vertices_map)
	return vertex_labels, node_part_label_vertices_map, variables, capacity_dict, capacities



def correct_cut_set(A,A_truth,cuts,single, unif_update):
	'''Correct each cut in cuts in A to match A_truth

	Returns corrected A and a list indicating if each cut in cuts was heavy or not with 1 or 0

	Args:
		A - probabilistic adjacency matrix 
		A_truth - adjacency matrix of template graph 
		cuts - list of lists of vertices 
		single - Boolean flag for whether or not to use single constraint.
			If False, use triple.
		unif_update - Boolean flag for wheter or not to use uniform update. 
			If false, use psuh update.
	'''
	vertex_labels, node_part_label_vertices_map, variables, capacity_dict, capacities = construct_node_partition(A_truth.shape[0],cuts)
	y_star = np.array(gen_assignment(A_truth,node_part_label_vertices_map,variables))
	y_prime = np.array(gen_assignment(A,node_part_label_vertices_map,variables))
	assert(len(variables) == 3**len(cuts))
	y, heavy = gen_opt_assignment(len(cuts), y_star, y_prime, capacities, single, variables) #optimize for new assignment
	diff = dict(zip(variables,y - y_prime))
	current = dict(zip(variables,y_prime))
	new_sol = dict(zip(variables,y))
	A_before = np.copy(A)
	'''Compute amount to add/remove from subgraphs defined by node partition pairs'''
	pair_weight_all, sub_graphs_all = sub_graph_assignment(A,diff,current,new_sol,node_part_label_vertices_map,variables,unif_update)
	'''Update subgraphs in matrix'''
	A_temp = update_sub_graphs(A, pair_weight_all, sub_graphs_all, node_part_label_vertices_map, unif_update)
	'''Correct for any errors in QP'''
	overflow = A_before.sum() - A_temp.sum()
	np.fill_diagonal(A_temp,1)
	A = update_helper(A_temp, overflow)
	np.fill_diagonal(A,0)
	return A, heavy


def gen_cuts(cut_params, num_cuts, A):
	'''Generate num_cuts specified by paramaters in cut_params

	Return list of cuts and their complements

	Args:
		cut_params - dictionary of cut sampling paramaters
		num_cuts - number of cuts to sample 
		A - probabilistic adjacency matrix 
	'''
	cut_gen = cut_params['opt']
	independent = cut_params['independent']
	cuts = []
	cuts_comp = []
	n = A.shape[0]
	if cut_gen == 'grasp':
		f = cut_params['f']
		f_global = cut_params['f_global']
		g = cut_params['g']
		frac_nodes_init = cut_params['frac_nodes_init']
		grasp_neighborhood_size = cut_params['grasp_neighborhood_size']
		assert(grasp_neighborhood_size <= 1.0)
		num_grasp_repeats = cut_params['num_grasp_repeats']
		neighborhood_size = int(n*grasp_neighborhood_size)
		out_graph, in_graph, out_neighbors, in_neighbors = sample_neighborhood_subset(n, neighborhood_size,A,A,[np.arange(n)]*n,n)
	elif cut_gen == 'goemmans-williams':
		G = nx.from_numpy_matrix(A)
	if independent:
		for i in range(num_cuts):
			if cut_gen == 'grasp':
				s = max(2,int(n*frac_nodes_init))
				c_set, c_comp_set = grasp(A,out_graph,s,num_grasp_repeats,f,f_global,g,out_neighbors,in_graph,in_neighbors,neighborhood_size)
				c = list(c_set)
				c_comp = list(c_comp_set)
			elif cut_gen == 'goemmans-williams':
				sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(G)
				c = list(sdp_cut.left)
				c_comp = list(sdp_cut.right)
			else:
				print(error_cut_gen_not_impl)
			cuts.append(c)
			cuts_comp.append(c_comp)
	else: #generate recursively
		partitions = []
		while (len(partitions) < num_cuts):
			if partitions == []:
				if cut_gen == 'grasp':
					c_set, c_comp_set = grasp(A,out_graph,int(n*frac_nodes_init),num_grasp_repeats,f,f_global,g,out_neighbors,in_graph,in_neighbors,neighborhood_size)
					partitions = [list(c_set),list(c_comp_set)]
				elif cut_gen == 'goemmans-williams':
					sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(G)
					partitions = [list(sdp_cut.left),list(sdp_cut.right)]
			else:
				partitions.sort(key=len,reverse=True)
				cur_partitions = len(partitions)
				num_to_divide = min(cur_partitions,num_cuts-cur_partitions)
				partitions_temp = partitions[num_to_divide:]
				for i in range(num_to_divide):
					nodes = list(partitions[i])
					B = A[nodes][:,nodes] #compute sub-graph and generate cut of subgraph
					if cut_gen == 'grasp':
						n_prime = B.shape[0]
						neighborhood_size_prime = int(n_prime*grasp_neighborhood_size)
						out_graph_prime, in_graph_prime, out_neighbors_prime, in_neighbors_prime = sample_neighborhood_subset(n_prime, neighborhood_size_prime,B,B,[np.arange(n_prime)]*n_prime,n_prime)
						c_set, c_comp_set = grasp(B,out_graph_prime,int(n_prime*frac_nodes_init),num_grasp_repeats,f,f_global,g,out_neighbors_prime,in_graph_prime,in_neighbors_prime,neighborhood_size_prime)
						left = [nodes[i] for i in list(c_set)]
						right = [nodes[i] for i in list(c_comp_set)]
					elif cut_gen == 'goemmans-williams':
						G = nx.from_numpy_matrix(B)
						sdp_cut = cvxgr.algorithms.goemans_williamson_weighted(G)
						left = [nodes[i] for i in list(sdp_cut.left)]
						right = [nodes[i] for i in list(sdp_cut.right)]
					partitions_temp.append(left)
					partitions_temp.append(right)
				partitions = partitions_temp
		cuts = cuts+partitions
		cuts_comp = [complement([c],n) for c in cuts]
	assert(len(cuts) == len(cuts_comp))
	assert(len(cuts) == num_cuts)
	return cuts, cuts_comp


def cut_sample_and_correct(A, A_truth, hyperparams, num_cuts = 1, grasp_neighborhood_size = 1.0, passed_cuts =[]):
	'''Sample num_cuts cuts to correct and correct them in A 

	Returns 
		probailistic matrix A with num_cuts cuts corrected
		list of bits for each cut that take 1 if cut sampled
			has sum of entries crossing larger than A_truth 
		list of cuts sampled 
		list of the complements of cuts sampled

	Args:
		A - proposal probabilistic adjacency matrix 
		A_truth - adjacency matrix template
		hyperparams
		num_cuts - number of cuts to sample before correcting (default 1)
		grasp_neighborhood_size - fraction of neighbors to use for grasp (default 1.0)
		passed_cuts - list of previously computed cuts (default is empty list)
	'''
	config = gen_cut_correct_config(hyperparams, grasp_neighborhood_size) 
	cuts, cuts_comp = gen_cuts(config, num_cuts-len(passed_cuts), A-A_truth) #sample cuts
	cuts = cuts+passed_cuts 
	cuts_comp = cuts_comp + [complement([c],n) for c in passed_cuts]
	cut_sets = [set(c) for c in cuts]
	assert(len(cuts) == num_cuts)
	A, heavy_list = correct_cut_set(A,A_truth, cut_sets,config['single'],config['unifup']) #correct cuts
	assert(len(heavy_list) == num_cuts)
	return A, heavy_list, cuts, cuts_comp

''' MAIN METHOD '''


def cut_correct_gen(path,num_cut_list,hyperparams,A_truth,A_proposal,cut_stats,grasp_neighborhood_size,batch_size=1,start_target=0,passed_cuts=[]):
	'''Sampe/correct cuts in A_proposal to match A_truth and write correct probabilistic 
	adjacency matrices.

	Returns list of probabilistic adjacency matrix statistics, 
		map of statistics to results
		and list of times it took to correct each cut (seconds).

	Args:
		path - file to write probabilistic adjacency matrices
		num_cut_list - list of number of cuts to correct before writing
		hyperparams - method for sampling/correcting
		A_truth - adjacency matrix template graph
		A_proposal - proposal probabilistic adjacency matrix 
		cut_stats - list of statistics to compute on each sampled cut 
		batch_size - number of cuts to sample before correcting (default is 1)
		start_target - number of cuts to correct before writing first 
			probabilistic adjacency matrix (default is 0)
		passed_cuts - list of cuts previously sampled to correct

	'''
	num_cut_list = sorted(num_cut_list) #number of cuts should be increasing
	results = defaultdict(dict) #map statistic name to map of cuts correct to value
	batch_cut_correct_stats = ['l2_lin_freq','entropies','cut_results']
	for x in batch_cut_correct_stats:
		results[x] = defaultdict(list)
	num_targets = len(num_cut_list)
	A = copy.deepcopy(A_proposal)
	true_spec = utils.spectrum(A_truth)
	num_corrections = 0
	cuts_over_all_targets = []
	assert(start_target <= num_targets)
	assert(start_target >= 0)
	times = []
	if start_target == 0: #if cuts corrected yet, record starting statistics
		if num_cut_list[0]>0:
			calls_to_correct = 0
			results['l2_lin_freq'][0] = [utils.l2_lin_weight(utils.spectrum(A),true_spec)]
			results['entropies'][0] = [np.mean(utils.entropy_m(A))]
	else: #get previous number of cuts corrected
		calls_to_correct = num_cut_list[start_target-1]
	for i in range(start_target,num_targets): 
		num_cuts_to_correct = int(num_cut_list[i])
		A_prev = A
		while calls_to_correct < num_cuts_to_correct:
			start_time = time.time()
			A_prev = np.copy(A)
			A, high, cuts, cuts_comp  = cut_sample_and_correct(A, A_truth, hyperparams, batch_size, grasp_neighborhood_size,passed_cuts)
			cuts_over_all_targets = cuts_over_all_targets+cuts
			calls_to_correct = calls_to_correct + batch_size
			assert(len(cuts)==batch_size)
			for i in range(len(cuts)): #record statistics on all cuts sampled
				update_cut_results(A_prev,cuts[i],cuts_comp[i],high[i],cut_stats,results['cut_results'],A_truth)
			times.append(time.time() - start_time)
		results['l2_lin_freq'][num_cuts_to_correct] = [utils.l2_lin_weight(utils.spectrum(A),true_spec)]
		results['entropies'][num_cuts_to_correct] = [np.mean(utils.entropy_m(A))]
		np.savetxt(path+'num_cuts_{}.txt'.format(num_cuts_to_correct), A)
	return batch_cut_correct_stats, results, times	
