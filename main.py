import time
import os
import random
import math

from helper_functions import *
from AUC_measures import *
import networkx as nx
import numpy as np

def predict_edges(G, train_graph, test_edges, index, num_trials, parameter = None):
	mat_score_indexs = ["cn", "lp"]
	extra_mat_score_indexs = ["jaccard", "lhn1", "salton", "sorensen", "hpi", "hdi"]
	sum_indices = ["aa", "ra", "ra_e", "ra_e2"]
	random_walker_indices = ["rw", "rwr"]
	experimental_indices = ["hpi_e", "hdi_e", "salton_e", "lhn1_e"]

	non_edges = n_random_non_edges(G, num_trials)
	selected_test_edges = random.choices(test_edges, k = len(non_edges))
	
	if index in mat_score_indexs or index in extra_mat_score_indexs:
		nodelist = [n for n in train_graph.nodes()]
		mat = nx.adjacency_matrix(train_graph, nodelist = nodelist)
		cn_mat = mat * mat

		if index in mat_score_indexs:
			if index == "cn":
				score_mat = cn_mat
			elif index == "lp":
				if parameter > 0 and parameter <= 1:
					score_mat = cn_mat + (parameter * (cn_mat * mat))
				else:
					raise ParameterError("Parameter, e,  must be 0 < e <= 1 for Local Path index")

			return mat_AUC_score(score_mat, selected_test_edges, non_edges, nodelist)

		elif index in extra_mat_score_indexs:
			return extra_mat_AUC_score(cn_mat, train_graph, selected_test_edges, nodelist, non_edges, index)

	elif index == "pa":
		return pa_AUC_score(train_graph, selected_test_edges, non_edges)

	elif index in sum_indices:
		return aa_ra_AUC_score(train_graph, selected_test_edges, non_edges, index, parameter)

	elif index in random_walker_indices:
		return rw_AUC_score(train_graph, selected_test_edges, non_edges, index)

	elif index in experimental_indices:
		nodelist = [n for n in train_graph.nodes()]
		mat = nx.adjacency_matrix(train_graph, nodelist = nodelist)
		cn_mat = mat * mat
		return experimental_AUC_score(train_graph, selected_test_edges, nodelist, cn_mat + (parameter * (cn_mat * mat)), non_edges, index)
	else:
		raise KeyError("Invalid index {} passed to predict_edges()".format(index))

def k_fold_train_and_test(G, k = 10, index = "cn", num_trials = 1000, parameter = None):
	#print("{}-fold cross validation of {} on graph with {} edges/{} nodes".format(k, index, nx.number_of_edges(G), nx.number_of_nodes(G)))
	#print("Splitting edges into test sets")
	subsets = k_set_edge_split(G, k)

	AUC_total = 0
	#print("Training")
	for test_edges in subsets:
		training_graph = G.copy()
		training_graph.remove_edges_from(test_edges)

		score = predict_edges(G.copy(), training_graph, test_edges, index, num_trials, parameter = parameter)

		AUC_total += score

	return AUC_total / k

def measure_index_performance(index, G, num_trials = 2000, num_repeats = 10, parameter = 0.01):
	total = 0
	for repeat in range(num_repeats):
		score = k_fold_train_and_test(G, index = index, num_trials = num_trials, parameter = parameter)
		total += score
	return total / num_repeats

file_output = True
all_indices = ["cn", "salton", "jaccard", "lhn1", "sorensen", "hpi", "hdi", "pa", "aa", "ra", "lp", "salton_e", "lhn1_e", "hpi_e", "hdi_e", "ra_e", "ra_e2"]
all_graphs = ["books", "netscience", "lastfm", "power", "condmat", "facebook", "movies"]
if __name__ == "__main__":
	for graph in ["books", "netscience", "power", "lastfm"]:
		if file_output:
			output_file = open(graph + "-out", "w")
		
		print(graph)
		G = load_graph(graph)
		for index in ["cn", "pa", "ra", "hpi_e"]:
			print(index)
			AUC = measure_index_performance(index, G)
			out = "{}\nAverage AUC: {:.4f}\n".format(index, AUC)
			print(out)
			if file_output:
				output_file.write(out)

		if file_output:
			output_file.close()
		print("--------------------")