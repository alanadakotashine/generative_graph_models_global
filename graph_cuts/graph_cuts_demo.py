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
import cut_gen
import rw_gen
import cut_walk_gen_utils


'''GLOBALS FOR PATH NAMES'''

def DATA_FILE(name):
	return '../data/'+name+'.txt'

def CONSOLIDATION_DIRECTORY(path, header):
	return path+header+'/'

def PARAMS_JSON_PATH(header):
	return header+'_params.json'

def STATS_JSON_PATH(header):
	return header+'_stats.json'

def NUM_WALK_ALG_ITERS_JSON_PATH(header):
	return header+'_num_walk_alg_iters.json'

def WALK_GEN_FREQ_PATH_HEAD_SINGLE_RUN(name, i, alg):
	return name+'_frequency+{}{}'.format(i,alg)

def WALK_GEN_PLOT_PATH_HEAD_SINGLE_RUN(name, i, alg):
	return name+'_'+alg+'_'+str(i)+'_'

def WALK_GEN_FREQ_PATHS_SINGLE_RUN(name,walk_algs, i):
	freq_paths = {}
	plot_paths = {}
	for alg in walk_algs:
		freq_paths[alg] = WALK_GEN_FREQ_PATH_HEAD_SINGLE_RUN(name, i, alg)
		plot_paths[alg] = WALK_GEN_PLOT_PATH_HEAD_SINGLE_RUN(name, i, alg)
	return freq_paths, plot_paths

def WALK_GEN_PLOT_PATH_HEAD_SINGLE_RUN_DATA(name,alg,i,prop):
	return WALK_GEN_PLOT_PATH_HEAD_SINGLE_RUN(name, i, alg) +prop+'.txt'


def WALK_GEN_FREQ_PATH_W_NUM_ITERS(name,alg,i,num_iters):
	return WALK_GEN_FREQ_PATH_HEAD_SINGLE_RUN(name, i, alg)+'num_iters{}.txt'.format(num_iters)


def CUT_CORRECT_PATH(path, name, i, cut_method):
	return path+name+'_num_cuts_{}_'.format(i)+cut_method

def CUT_CORRECT_W_WALK_ALG_SM(path, name, i, cut_method, freq, num_walk_iters, cuts_corrected):
	return CUT_CORRECT_W_WALK_ALG_SM_HEADER(path, name, i, cut_method, freq, num_walk_iters) + 'num_cuts_{}.txt'.format(cuts_corrected)

def CUT_CORRECT_W_WALK_ALG_SM_RAW(path, name, i, cut_method, freq, num_walk_iters, cuts_corrected):
	return CUT_CORRECT_W_WALK_ALG_SM_HEADER('', name, i, cut_method, freq, num_walk_iters) + 'num_cuts_{}.txt'.format(cuts_corrected)

def CUT_CORRECT_HEADER_ONE_RUN(path, header, i, cut_method):
	return path+header+'_num_cuts_{}_'.format(i)+cut_method

def CUT_CORRECT_W_WALK_ALG_HEADER(path, header, i, cut_method, freq_gen, num_walk_alg_iter):
	return CUT_CORRECT_HEADER_ONE_RUN(path, header, i, cut_method) + '_'+freq_gen+'_{}'.format(num_walk_alg_iter)

def CUT_CORRECT_W_WALK_ALG_SM_HEADER(path, header, i, cut_method, freq_gen, num_walk_alg_iter):
	return CUT_CORRECT_W_WALK_ALG_HEADER(path, header, i, cut_method, freq_gen, num_walk_alg_iter) + '_sM'

def TIME_TO_CORRECT(path, header, i, cut_method, freq_gen, num_walk_alg_iter):
	return CUT_CORRECT_W_WALK_ALG_HEADER(path, header, i, cut_method, freq_gen, num_walk_alg_iter)+'time_to_correct.txt'

def TOTAL_TIME(path, header, i, cut_method, freq_gen, num_walk_alg_iter):
	return CUT_CORRECT_W_WALK_ALG_HEADER(path, header, i, cut_method, freq_gen, num_walk_alg_iter)+'total_time.txt'

def TIME_GEN_FREQ(path, header, i, freq_gen, num_walk_alg_iter):
	return WALK_GEN_PLOT_PATH_HEAD_SINGLE_RUN(path+header, i, freq_gen)+'{}_total_time.txt'.format(num_walk_alg_iter)


def CUT_CORRECT_DATA_JSON(path, header, i, cut_method, key, freq_gen, num_iter):
	return CUT_CORRECT_DATA_JSON_HEAD(path, header, i, cut_method, key, freq_gen, num_iter) +'.json'

def CUT_CORRECT_DATA_JSON_HEAD(path, header, i, cut_method, key, freq_gen, num_iter):
	return CUT_CORRECT_HEADER_ONE_RUN(path, header, i, cut_method)+'_'+key+'_'+freq_gen+'_{}'.format(num_iter)

def CUT_CORRECT_DATA_CONSOLIDATION_HEAD(path, header, cut_method, key, freq_gen, num_iter):
	return path+header+'_num_cuts_'+cut_method+'_'+key+'_'+freq_gen+'_{}'.format(num_iter)

def cut_correct_set_up(total_time, non_walk_gen, walk_algs, num_walk_alg_iters, cut_correct_methods, params, A_truth, i, name, path):
	num_walk_alg_iters_all = {} #map frequency generator to all walk algorithm iterations (including non walk algorithms to zero)
	#For the frequency matrices without walks, only key is zero for zero walk alg iterations
	frequencies_non_walk = cut_walk_gen_utils.gen_freq_non_walk(A_truth, non_walk_gen)
	for cut_method in cut_correct_methods: #For each cut method, write the frequency matrices for input so they can be loaded
		for freq_gen in non_walk_gen:
			np.savetxt(CUT_CORRECT_W_WALK_ALG_SM(path, params['header'][0], i, cut_method, freq_gen, 0, 0),frequencies_non_walk[freq_gen][0])
			num_walk_alg_iters_all[freq_gen] = [0]
		for freq_gen in walk_algs:
			for num_iter in num_walk_alg_iters[freq_gen]:
				freq_path = WALK_GEN_FREQ_PATH_W_NUM_ITERS(path+params['header_short'][0],freq_gen,i,num_iter)
				F = np.loadtxt(freq_path)
				np.savetxt(CUT_CORRECT_W_WALK_ALG_SM(path,params['header'][0], i, cut_method, freq_gen, num_iter, 0),F)
			num_walk_alg_iters_all[freq_gen] = num_walk_alg_iters[freq_gen]
	for cut_method in cut_correct_methods:
		for non_walk_alg in non_walk_gen:
			total_time[cut_method][non_walk_alg+str(0)] = 0.0
		for walk_alg in walk_algs:
			for num_iter in num_walk_alg_iters[walk_alg]:
				time_gen_freq = np.loadtxt(TIME_GEN_FREQ(path, params['header'][0], i, walk_alg, num_iter))
				total_time[cut_method][walk_alg+str(num_iter)]=float(time_gen_freq)
	all_gens = walk_algs + non_walk_gen
	return num_walk_alg_iters_all, all_gens, total_time

def walk_alg_w_cut_correct(path,A_truth, stats, params,i, num_walk_alg_iters = None):
	'''Walk algorithms

	path -- write the new probabilistic adjacency matirces generated.



	'''
	frequencies = {}
	walk_algs = params['walk_algs']
	non_walk_gen = params['non_walk_gen']
	start_target = int(params['start_target'][0])
	cut_correct_methods = params['cut_correct_methods']
	cut_correct_header = params['header'][0]
	'''total time maps the cut correction method to a dictionary that maps
	the frequency matrix name to the total time it takes
	to correct each of the frequency matrixes it corrects, including
	the time it takes to construct the frequency matrix corrected'''
	total_time = defaultdict(dict)
	#initializes the dictionaries for each cut method
	for cut_method in cut_correct_methods:
		total_time[cut_method] = defaultdict(float)
	if params['run_walk_algs'][0]==1:
		freq_paths, data_paths = WALK_GEN_FREQ_PATHS_SINGLE_RUN(path+params['header_short'][0], walk_algs, i)
		time_walks = rw_gen.rw_gen(A_truth,walk_algs,num_walk_alg_iters,freq_paths, data_paths)
		#record time for every cut_method
		for alg in walk_algs:	
			for num_iter in num_walk_alg_iters[alg]:
				'''Time was recorded for the time it took to take num_iter walk iterations for each alg '''
				np.savetxt(TIME_GEN_FREQ(path, params['header'][0], i, alg, num_iter),[time_walks[alg][num_iter]])
	'''Cut correct'''
	#If resuming correcting cuts, make sure that we already ran any walk algoirthms
	if start_target > 0:
		assert(params['run_walk_algs'][0] == 0)
	num_walk_alg_iters_all, all_gens, total_time = cut_correct_set_up(total_time, non_walk_gen, walk_algs, num_walk_alg_iters, cut_correct_methods, params, A_truth, i, cut_correct_header, path)
	num_cut_stops = params['num_cut_stops']
	cut_correct_batch_size = params['cut_correct_batch_size'][0]
	grasp_neighborhood_size = params['grasp_neighborhood_size'][0]
	'''results maps a statistic to a map of cuts corrected to the value that
	statistic took for the model for that number of cuts corrected
	results{stat_name -> {cuts_corrected: [stat_value]}}'''
	results = defaultdict(dict)
	for cut_method in cut_correct_methods:
		#maps the name of each frequency matrix input to list of times, 
		#ith item in list is time it took to correct ith cut
		time_to_correct = defaultdict(list)
		for freq_gen in all_gens:
			for num_iter in num_walk_alg_iters_all[freq_gen]:
				#Initialize time or load
				if start_target == 0:
					calls_corrected = 0
					time_to_correct_ = []
				else:
					#load
					time_to_correct_ = list(np.loadtxt(TIME_TO_CORRECT(path, params['header'][0], i, cut_method, freq_gen, num_iter)))
					#store in dictionaries
					time_to_correct[freq_gen+str(num_iter)] = time_to_correct_
					calls_corrected = num_cut_stops[start_target-1]
					total_time_ = float(np.loadtxt(TOTAL_TIME(path, params['header'][0], i, cut_method, freq_gen, num_iter)))
					total_time[cut_method][freq_gen+str(num_iter)] = total_time_
				#Load files to correct
				sm_file_head = CUT_CORRECT_W_WALK_ALG_SM_HEADER(path, params['header'][0], i, cut_method, freq_gen, num_iter)
				input_to_correct = np.loadtxt(CUT_CORRECT_W_WALK_ALG_SM(path, params['header'][0], i, cut_method, freq_gen, num_iter, calls_corrected))
				'''Returns map of result from name of statistic to a map'''
				'''Each statistic map is a map from the number of cuts corrected to that
				statistic on the frequency matrix'''
				result_keys, results, time_to_correct_new = cut_gen.cut_correct_gen(sm_file_head,num_cut_stops,cut_method,A_truth,input_to_correct,stats['cut'],grasp_neighborhood_size,cut_correct_batch_size,start_target)
				'''Append the time it took to correct the new cuts'''
				time_to_correct[freq_gen+str(num_iter)] = time_to_correct_ + time_to_correct_new
				'''Add total time it took to correct the new cuts'''
				total_time[cut_method][freq_gen+str(num_iter)] += sum(time_to_correct_new)
				for key in result_keys:
					json_data = CUT_CORRECT_DATA_JSON(path, params['header'][0], i, cut_method, key, freq_gen, num_iter)
					if os.path.exists(json_data):
						with open(json_data,'r') as fp:
							results_old = json.load(fp)
						if key != 'cut_results':
							results_old = {int(k):v for k,v in results_old.items()}
						for k in results_old:
							if k not in results[key]:
								results[key][k] = results_old[k]
					#Once statistic is updated, dump results into file name
					with open(json_data,'w') as fp:
						json.dump(results[key],fp)
				time_to_correct_ = time_to_correct[freq_gen+str(num_iter)]
				time_all_ = total_time[cut_method][freq_gen+str(num_iter)]
				np.savetxt(TIME_TO_CORRECT(path, params['header'][0], i, cut_method, freq_gen, num_iter),np.array(time_to_correct_))
				np.savetxt(TOTAL_TIME(path, params['header'][0], i, cut_method, freq_gen, num_iter),np.array([time_all_]))


	
def consolidate_data_across_runs(name,path,stats,params, check_point_header,props,num_walk_alg_iters,directory=''):
	#consolidate the results results for each cut method
	freq_mat_nums_all = params['freq_mat_nums_all']
	walk_algs = params['walk_algs']
	non_walk_gen = params['non_walk_gen']
	results = defaultdict(dict)
	cut_methods = params['cut_correct_methods']
	cuts_corrected = params['num_cut_stops']
	all_freq_gens = walk_algs + non_walk_gen
	for gen in non_walk_gen:
		num_walk_alg_iters[gen]=[0]
	for cut_method in cut_methods:
		for freq_gen in all_freq_gens:
			for num_iter in num_walk_alg_iters[str(freq_gen)]:
				specs_initial = []
				specs_final = []
				#Loop through all iterations for cut_method and freq_gen constructed from num_iter
				#walk alg iterations
				for i in freq_mat_nums_all:
					#graph_head = 
					#write initial and final graphs into directory
					final = max(cuts_corrected)
					js = [0,5,10,final]
					for j in js:
						graph = CUT_CORRECT_W_WALK_ALG_SM_RAW(path, name, i, cut_method, freq_gen, num_iter,j)
						if os.path.exists(path+graph):
							A = np.loadtxt(path+graph)
							np.savetxt(directory+graph,A)
				#For each property in props, loop through all iterations to get json and compute
				#mean and standard deviation for that property
				for d in props:
					data_total = defaultdict(list)
					for i in freq_mat_nums_all:
						#load the data
						json_data_path = CUT_CORRECT_DATA_JSON(path, name, i, cut_method, d, freq_gen, num_iter)
						if os.path.exists(json_data_path):
							with open(json_data_path,'r') as fp:
								data = json.load(fp)
							#move data to directory
							with open(directory+CUT_CORRECT_DATA_JSON('', name, i, cut_method, d, freq_gen, num_iter),'w') as fp:
								json.dump(data,fp)
							print(data)
						else:
							with open(directory+json_data_path,'r') as fp:
								data = json.load(fp)
						#maps for cut_results are lists, for all others are floats
						for key in data:
							if d == 'cut_results':
								data_total[key] = data_total[key]+data[key]
							else:
								data_total[key].append(data[key])
					if d != 'cut_results':
						json_head = CUT_CORRECT_DATA_CONSOLIDATION_HEAD('', params['header'][0], cut_method, d, freq_gen, num_iter)
						cut_walk_gen_utils.record_dict_mean_std(data_total,json_head,directory)
					else:					
						for stat in stats['cut']:
							txt_head = CUT_CORRECT_DATA_CONSOLIDATION_HEAD('', params['header'][0], cut_method, stat, freq_gen, num_iter)
							np.savetxt(directory+txt_head,data_total[stat])
					
def main(path, params_path, stats_path, num_walk_alg_iters_path, combine):
	'''
	path -- 
	params_path -- path with dictionary of paramaters 
	stas_path -- path with dictionary of statistics to run on probabilisitc
		adjacency matrices 
	num_walk_alg_iters_path -- number of walk algorithm iterations for random walk 
		algorithms 
	combine -- flag. True if taking average over past runs, False if executing
		new run.
	'''
	with open(params_path,'r') as fp:
		params = json.load(fp)
	with open(stats_path,'r') as fp:
		stats = json.load(fp)
	with open(num_walk_alg_iters_path,'r') as fp:
		num_walk_alg_iters = json.load(fp)
	model_stats = ['entropies','l2_lin_freq']
	if combine == 0: #generate model using walks/cut correct 
		A_truth = np.loadtxt(DATA_FILE(name))
		np.fill_diagonal(A_truth,0) #remove self loops
		freq_mat_nums_compute = params['freq_mat_nums_compute']
		for i in freq_mat_nums_compute:
			walk_alg_w_cut_correct(path,A_truth, stats, params,i,num_walk_alg_iters)
	else:
		cut_correct_batch_size = params['cut_correct_batch_size'][0]
		grasp_neighborhood_size = params['grasp_neighborhood_size'][0]
		header = params['header'][0]
		consolidation_directory = CONSOLIDATION_DIRECTORY(path,header)
		try:
			os.mkdir(consolidation_directory[:-1])
		except OSError:
			print('{} directory exists'.format(header))
		params_path = PARAMS_JSON_PATH(header)
		stats_path = STATS_JSON_PATH(header)
		num_walk_alg_iterations_path = NUM_WALK_ALG_ITERS_JSON_PATH(header)
		with open(consolidation_directory+params_path,'w') as fp:
			json.dump(params,fp)
		with open(consolidation_directory+stats_path,'w') as fp:
			json.dump(stats,fp)
		with open(consolidation_directory+num_walk_alg_iterations_path,'w') as fp:
			json.dump(num_walk_alg_iters,fp)
		'''Move cut correct data over'''
		consolidate_data_across_runs(header,path,stats,params, 'num_cuts', model_stats+['cut_results'],num_walk_alg_iters,consolidation_directory)
		'''Move random walk data over'''
		walk_algs = params['walk_algs']
		for alg in walk_algs:
			for iteration in params['freq_mat_nums_all']:
				for prop in ['conductance','checkpoints','l2_lin','entropy','num_rounds']:
					src = WALK_GEN_PLOT_PATH_HEAD_SINGLE_RUN_DATA(name,alg,iteration,prop)
					if os.path.exists(path+src):
						shutil.copyfile(path+src, consolidation_directory+src)
				for walk_alg_iter in num_walk_alg_iters[alg]:
					src = WALK_GEN_FREQ_PATH_W_NUM_ITERS(name,alg,iteration,walk_alg_iter)
					if os.path.exists(path+src):
						shutil.copy(path+src,consolidation_directory+src)
		'''Move time files over'''
		for i in params['freq_mat_nums_all']:
			for cut_method in params['cut_correct_methods']:
				for freq_gen in params['walk_algs']+params['non_walk_gen']:
					for num_iter in num_walk_alg_iters[freq_gen]:
						time_to_correct_file = TIME_TO_CORRECT('', params['header'][0], i, cut_method, freq_gen, num_iter)
						total_time_file = TOTAL_TIME('', params['header'][0], i, cut_method, freq_gen, num_iter)
						src_correct = path + time_to_correct_file 
						dst_correct = consolidation_directory + time_to_correct_file
						shutil.copy(src_correct, dst_correct)
						src_total = path + total_time_file 
						dst_total = consolidation_directory + total_time_file
						shutil.copy(src_total, dst_total)

if __name__ == "__main__":
	#cut_gen.test_gen_cuts()
	#cut_gen.test_correct_balance_batch()
	#cut_gen.test_opt_assignment()
	#cut_gen.test_gen_cuts()
	name = sys.argv[1]
	path = sys.argv[2]
	stats_path = sys.argv[3]
	params_path = sys.argv[4]
	num_walk_alg_iters = sys.argv[5]
	combine = int(sys.argv[6])
	main(path, params_path,stats_path, num_walk_alg_iters,combine)	
