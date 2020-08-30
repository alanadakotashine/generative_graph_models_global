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


'''Utils'''

def gen_cluster_graph(num_clusters,cluster_size,p,q):
	'''Generate synthetic graph with num_clusters of size cluter_size.'''
	'''Intra-cluster edge probability p, inter-cluster edge prob q'''
	n = num_clusters*cluster_size
	A = np.zeros((n,n))
	for i in range(num_clusters):
		#generate cluster
		P = nx.adjacency_matrix(nx.gnp_random_graph(cluster_size,p)).todense()
		A[i*cluster_size:(i+1)*cluster_size,i*cluster_size:(i+1)*cluster_size] = P
		for j in range(i+1,num_clusters):
			Q = nx.adjacency_matrix(nx.gnp_random_graph(cluster_size,q)).todense()
			A[i*cluster_size:(i+1)*cluster_size,j*cluster_size:(j+1)*cluster_size] = Q
			A[j*cluster_size:(j+1)*cluster_size,i*cluster_size:(i+1)*cluster_size] = Q
	return A

def random_convex_comb(vs):
	n = len(vs)
	x = sorted([random.random() for i in range(n)])
	alpha = [x[0]] + [x[i+1]-x[i] for i in range(n-1)]
	result = np.zeros(len(vs[0]))
	for i in range(n):
		weight = [alpha[i]*el for el in vs[i]]
		result = result + weight
	return result


def complement(sets,n):
	'''Compute all the nodes in range(n) that are not in the union of sets'''
	comp = []
	#placed set is the union of all sets in sets
	placed_set = set()
	for s in sets:
		placed_set = placed_set.union(s)
	#for each node i in range(n) not placed, add to comp
	for i in range(n):
		in_set = False
		if i not in placed_set:
			comp.append(i)
	assert(n == len(placed_set) + len(comp))
	return comp


def lower_ones(A):
	return np.triu(A,1) + np.tril(np.ones_like(A),0)

def lower_zeros(A):
	return np.triu(A,1) + np.tril(np.zeros_like(A),0)



def absolute_difference(x,y):
	Z = zip(x,y)
	acc = 0
	for (i,j) in Z:
		acc = acc + np.abs((i-j))
	return acc



def comp_graph_stats(A,stats):
	data = defaultdict(list)
	G = nx.from_numpy_matrix(A)
	for stat in stats:
		if stat == 'spectrum':
			data[stat] = utils.spectrum(A)
		if stat == 'size_lcc':
			data[stat] = [utils.size_lcc(A,G)]
		elif stat == 'betweenness':
			data[stat] = utils.betweenness(A,G)
		elif stat == 'shortestpath':
			data[stat] = utils.shortestpath(A,G)
		elif stat == 'cc':
			data[stat] = utils.cc(A,G)
		elif stat == 'assort':
			data[stat] = [utils.assort(A,G)]
	return data

def identity(x):
	return x 

def neg_identity(x):
	return -1*x


'''SBM benchmark'''

def gen_sbm_entropy(A,max_k,plot_path):
	entropy = []
	l2_lins = []
	true_spec = utils.spectrum(A)
	for k in range(2,max_k+1):
		sbm_labels = spectral_clustering(A, n_clusters=k, eigen_solver='arpack')
		sbm_P = gen_sbm_from_labels(k,sbm_labels,A)
		entropy.append(utils.entropy_m_sum(sbm_P))
		l2_lins.append(utils.l2_lin_weight(utils.spectrum(sbm_P),true_spec))
	plt.plot(range(2,k+1),entropy)
	plt.savefig(plot_path+'_sbm_entropy.pdf')
	plt.xlabel('Num Clusters')
	plt.ylabel('Entropy')
	plt.gcf().clear()
	np.savetxt(plot_path+'_sbm_entropies.txt',entropy)
	plt.plot(range(2,k+1),l2_lins)
	plt.savefig(plot_path+'_sbm_l2_lins.pdf')
	plt.xlabel('Num Clusters')
	plt.ylabel('Spectra L2 Norm Linear Weights')
	plt.gcf().clear()
	np.savetxt(plot_path+'_sbm_l2_lins.txt',l2_lins)

def gen_sbm_from_labels(k,labels,A):
	n = len(labels)
	P = np.zeros((n,n))
	for i in range(k):
		idx_i = [node for node in range(n) if labels[node]==i]
		num_nodes_cluster = len(idx_i)
		cluster = A[np.ix_(idx_i,idx_i)]
		num_edges = cluster.sum()
		cluster_size = (num_nodes_cluster**2) - num_nodes_cluster
		p = float(num_edges)/float(cluster_size)
		cluster_prob = np.ones((num_nodes_cluster,num_nodes_cluster))-np.eye(num_nodes_cluster)
		P[np.ix_(idx_i,idx_i)] = p*cluster_prob
		for j in range(i+1,k):
			idx_j = [node for node in range(n) if labels[node]==j]
			num_nodes_cluster_j = len(idx_j)
			cross_cluster = A[np.ix_(idx_i,idx_j)]
			num_edges = cross_cluster.sum()
			cross_size = num_nodes_cluster*num_nodes_cluster_j
			p = float(num_edges)/float(cross_size)
			cross_cluster_prob = np.ones((num_nodes_cluster,num_nodes_cluster_j))
			P[np.ix_(idx_i,idx_j)] = p*cross_cluster_prob
			P[np.ix_(idx_j,idx_i)] = p*np.transpose(cross_cluster_prob)
	return P


'''Non random benchmarks'''

def load_netgan_std_walk(name,truth_spectrum=None):
	freq_mat_paths = []
	entropy_target_placeholder = 0
	if truth_spectrum is not None:
		l2_lin_freq = {entropy_target_placeholder:[]}
		entropies = {entropy_target_placeholder:[]}
	for i in range(1,11):
		F = np.loadtxt('../netgan_64_g40_d30/'+name+'/'+name+'_rw_s64_g40_d30_{}'.format(i)+'/auc+ap_stop_model_scaledM.txt')
		freq_mat_paths.append(F)
		if truth_spectrum is not None:
			l2_lin_freq[entropy_target_placeholder].append(utils.l2_lin_weight(truth_spectrum, utils.spectrum(F)))
			entropies[entropy_target_placeholder].append(utils.entropy_m_sum(F))
	if truth_spectrum is not None:
		record_dict_mean_std(l2_lin_freq,name+'_l2_lin_freq_netgan')
		record_dict_mean_std(entropies,name+'_entropy_netgan')
	return freq_mat_paths

def gen_freq_non_walk(A_truth, algs,powers={}):
	frequencies_alg = {}
	n = A_truth.shape[0]
	for alg in algs:
		frequencies_alg[alg] = {}
		if alg == 'unif':
			frequencies_alg[alg][0] = utils.normMatrix_wsum(np.ones((n,n))-np.eye(n),A_truth.sum())
		elif alg == 'zero':
			frequencies_alg[alg][0] =  np.zeros((n,n))
		elif alg == 'ct':
			D = np.diag(A_truth.sum(axis=0))
			L = D-A_truth
			M_temp = np.linalg.pinv(L)
			if 'ct' in powers:
				power = powers['ct']
			else:
				power = 1
			frequencies_alg[alg][0] = freq_from_commute_time(A_truth,M_temp,power)
		elif alg == 'sp':
			SP = gen_sp(A_truth)
			if 'sp' in powers:
				power = powers['sp']
			else:
				power = 1
			frequencies_alg[alg][0] = freq_from_inverse(A_truth,SP,power)
	return frequencies_alg

def freq_from_commute_time(A,M_temp,k=1):
	n = A.shape[0]
	M = np.zeros((n,n))
	for u in range(n):
		for v in range(n):
			if (M_temp[u,u]+M_temp[v,v]-M_temp[u,v]-M_temp[v,u]) != 0 and u!=v:
				M[u,v] = 1.0/(((M_temp[u,u]+M_temp[v,v]-M_temp[u,v]-M_temp[v,u])*A.sum())**k)
	return M

def freq_from_inverse(A,S_temp,k=1):
	n = A.shape[0]
	S = np.zeros((n,n))
	for u in range(n):
		for v in range(n):
			if S_temp[u,v] != 0:
				S[u,v] = 1.0/(float(S_temp[u,v])**k)
	return S


def gen_sp(A):
	G = nx.from_numpy_matrix(A)
	nodes = G.nodes()
	n = A.shape[0]
	F = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			dist = nx.shortest_path_length(G,source=i,target=j)
			F[i,j] = dist
			F[j,i] = dist 
	return F




'''Data IO/Recording'''


def record_dict_mean_std(results,title,directory=''):
	print('recording mean std json')
	print(title)
	print(results)
	means = {}
	stds = {}
	data = {}
	for i in results:
		if results[i] != []:
			means[i] = np.mean(results[i])
			stds[i] = np.std(results[i])
			data[i] = results[i]
	with open(directory+title+'_data.json','w') as fp:
		json.dump(data,fp)
	with open(directory+title+'_mean.json','w') as fp:
		json.dump(means,fp)
	with open(directory+title+'_std.json','w') as fp:
		json.dump(stds,fp)






'''CONFIGS FOR ALGORITHMS'''

def algs_to_abvs(algs):
	abvs = {}
	abvs['baseline']='baseline'
	abvs['disc']='D'
	abvs['vote_20+d']='v20+D'
	abvs['vote_20+d+wdw']='v20+D+WDW'
	abvs['vote_5+d']='v5+D'
	abvs['vote_5+d+wdw']='v5+D+WDW'
	abvs['walk_distance_weight']='WDW'
	abvs['disc+wdw']='D+WDW'
	abvs['vote_20']='v20'
	abvs['vote_20+wdw']='v20+WDW'
	abvs['vote_5+wdw']='v5+WDW'
	abvs_algs = []
	for alg in algs:
		abvs_algs.append(abvs[alg])
	return abvs_algs













	
