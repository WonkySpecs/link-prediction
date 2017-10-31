import time
import os
import random
import math

from helper_functions import *
import networkx as nx
import scipy as sp
import numpy as np

#For indices whose scores can be determined with matrix calculations, it is viable to
#find the scores of all edges.
def mat_AUC_score(score_mat, test_edges, non_edges, nodelist):
	total = 0
	missing_edges = random.sample(test_edges, len(non_edges))
	for i in range(len(non_edges)):
		missing_edge = missing_edges[i]
		non_edge = non_edges[i]

		non_edge_score = score_mat[nodelist.index(non_edge[0]), nodelist.index(non_edge[1])]
		missing_edge_score = score_mat[nodelist.index(missing_edge[0]), nodelist.index(missing_edge[1])]

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5

	return total / float(len(non_edges))

def predict_edges(G, train_graph, test_edges, method, num_trials, parameter = None):
	if len(test_edges) < num_trials:
		raise ParameterError("Number of test edges ({}) must at least equal the number of trials ({}) for predict_edges()".format(len(test_edges), num_trials))

	mat_score_methods = ["cn", "lp"]
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

		return mat_AUC_score(score_mat, test_edges, n_random_non_edges(G, num_trials), nodelist)

	elif method == "jaccard":
		return jaccard_AUC(G, cn_mat, train_graph, test_edges, nodelist, num_trials)
	elif method == "lhn1":
		return lhn1_AUC(G, cn_mat, train_graph, test_edges, nodelist, num_trials)
	else:
		raise KeyError("Invalid method {} passed to predict_edges()".format(method))

def k_fold_train_and_test(G, k = 10, method = "cn", num_trials = 1000, parameter = None):
	print("{}-fold cross validation of {} on graph with {} edges/{} nodes".format(k, method, nx.number_of_edges(G), nx.number_of_nodes(G)))
	print("Splitting edges into test sets")
	subsets = k_set_edge_split(G, k)

	AUC_total = 0
	print("Training")
	for test_edges in subsets:
		training_graph = G.copy()
		training_graph.remove_edges_from(test_edges)

		score = predict_edges(G.copy(), training_graph, test_edges, method, num_trials, parameter = parameter)

		AUC_total += score

	return AUC_total / k

#TODO: Combine lhn1 and jaccard (and as yet unimplemented indices) as there is a ton of code reuse here
def lhn1_AUC(G, cn_mat, train_edges, test_edges, nodelist, num_trials):
	orig_G = G.copy()

	non_edges = n_random_non_edges(orig_G, num_trials)

	total = 0

	for non_edge in non_edges:
		missing_edge = random.sample(test_edges, 1)[0]

		u = nodelist.index(non_edge[0])
		v = nodelist.index(non_edge[1])
		non_edge_score = cn_mat[u, v] / len(orig_G[non_edge[0]]) * len((orig_G[non_edge[1]]))

		u = nodelist.index(missing_edge[0])
		v = nodelist.index(missing_edge[1])
		missing_edge_score = cn_mat[u, v] / len(orig_G[missing_edge[0]]) * len((orig_G[missing_edge[1]]))

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5
		test_edges.remove(missing_edge)

	return total / float(num_trials)

def jaccard_AUC(G, cn_mat, train_edges, test_edges, nodelist, num_trials):
	orig_G = G.copy()

	non_edges = n_random_non_edges(orig_G, num_trials)

	total = 0

	for non_edge in non_edges:
		missing_edge = random.sample(test_edges, 1)[0]

		u = nodelist.index(non_edge[0])
		v = nodelist.index(non_edge[1])
		non_edge_score = cn_mat[u, v] / len(set(orig_G[non_edge[0]]) | set(orig_G[non_edge[1]]))

		u = nodelist.index(missing_edge[0])
		v = nodelist.index(missing_edge[1])
		missing_edge_score = cn_mat[u, v] / len(set(orig_G[missing_edge[0]]) | set(orig_G[missing_edge[1]]))

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5
		test_edges.remove(missing_edge)

	return total / float(num_trials)

if __name__ == "__main__":
	G = load_graph("condmat")

	for i in [0.1, 0.2, 0.3, 0.4]:
		jaccard(G.copy(), i)

	for test_edge_fraction in [0.1, 0.4]:
		cn_train_and_AUC(G.copy(), test_edge_fraction)
		for e in [0.15 * n for n in range(1, 3)]:
			lp_train_and_AUC(G.copy(), test_edge_fraction, e)