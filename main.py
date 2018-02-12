import time
import os
import random
import math

from helper_functions import *
from AUC_measures import *
import networkx as nx
import numpy as np

def predict_edges(G, train_graph, test_edges, method, num_trials, parameter = None):
	mat_score_methods = ["cn", "lp"]
	extra_mat_score_methods = ["jaccard", "lhn1", "salton", "sorensen", "hpi", "hdi"]
	sum_indices = ["aa", "ra", "ra_e", "ra_e2"]
	random_walker_indices = ["rw", "rwr"]
	experimental_indices = ["hpi_e", "hdi_e", "salton_e", "lhn1_e"]

	non_edges = n_random_non_edges(G, num_trials)
	selected_test_edges = random.choices(test_edges, k = len(non_edges))
	
	if method in mat_score_methods or method in extra_mat_score_methods:
		nodelist = [n for n in train_graph.nodes()]
		mat = nx.adjacency_matrix(train_graph, nodelist = nodelist)
		cn_mat = mat * mat

		if method in mat_score_methods:
			if method == "cn":
				score_mat = cn_mat
			elif method == "lp":
				if parameter > 0 and parameter <= 1:
					score_mat = cn_mat + (parameter * (cn_mat * mat))
				else:
					raise ParameterError("Parameter, e,  must be 0 < e <= 1 for Local Path index")

			return mat_AUC_score(score_mat, selected_test_edges, non_edges, nodelist)

		elif method in extra_mat_score_methods:
			return extra_mat_AUC_score(cn_mat, train_graph, selected_test_edges, nodelist, non_edges, method)

	elif method == "pa":
		return pa_AUC_score(train_graph, selected_test_edges, non_edges)

	elif method in sum_indices:
		return aa_ra_AUC_score(train_graph, selected_test_edges, non_edges, method, parameter)

	elif method in random_walker_indices:
		return rw_AUC_score(train_graph, selected_test_edges, non_edges, method)

	elif method in experimental_indices:
		nodelist = [n for n in train_graph.nodes()]
		mat = nx.adjacency_matrix(train_graph, nodelist = nodelist)
		cn_mat = mat * mat
		return experimental_AUC_score(train_graph, selected_test_edges, nodelist, cn_mat + (parameter * (cn_mat * mat)), non_edges, method)
	else:
		raise KeyError("Invalid method {} passed to predict_edges()".format(method))

def k_fold_train_and_test(G, k = 10, method = "cn", num_trials = 1000, parameter = None):
	#print("{}-fold cross validation of {} on graph with {} edges/{} nodes".format(k, method, nx.number_of_edges(G), nx.number_of_nodes(G)))
	#print("Splitting edges into test sets")
	subsets = k_set_edge_split(G, k)

	AUC_total = 0
	#print("Training")
	for test_edges in subsets:
		training_graph = G.copy()
		training_graph.remove_edges_from(test_edges)

		score = predict_edges(G.copy(), training_graph, test_edges, method, num_trials, parameter = parameter)

		AUC_total += score

	return AUC_total / k

file_output = False
all_indices = ["cn", "salton", "jaccard", "lhn1", "sorensen", "hpi", "hdi", "pa", "aa", "ra", "lp", "salton_e", "lhn1_e", "hpi_e", "hdi_e", "ra_e", "ra_e2"]
all_graphs = ["books", "netscience", "lastfm", "power", "condmat", "facebook", "movies"]
if __name__ == "__main__":
	repeats = 1
	for graph in all_graphs:
		if file_output:
			output_file = open(graph + "-out", "w")
		
		print(graph)
		G = load_graph(graph)
		for method in ["cn", "ra", "ra_e", "ra_e2"]:
			total = 0
			print(method)
			start = time.clock()
			for i in range(repeats):
				score = k_fold_train_and_test(G.copy(), method = method, num_trials = 1000, parameter = 0.02)
				total += score
			out = "Average AUC: {:.4f}\nAverage time: {:.4f}\n".format(total / repeats, (time.clock() - start) / repeats)
			if file_output:
				output_file.write(method + "\n")
				output_file.write(out)
			print(out)

		if file_output:
			output_file.close()
		print("--------------------")