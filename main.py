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

#These indices require more processing than just looking up matrix elements
def extra_mat_AUC_score(G, cn_mat, train_graph, test_edges, nodelist, num_trials, index):
	non_edges = n_random_non_edges(G, num_trials)

	total = 0

	for non_edge in non_edges:
		missing_edge = random.sample(test_edges, 1)[0]

		u_non = nodelist.index(non_edge[0])
		v_non = nodelist.index(non_edge[1])

		u_miss = nodelist.index(missing_edge[0])
		v_miss = nodelist.index(missing_edge[1])

		if index == "jaccard":
			non_edge_score = cn_mat[u_non, v_non] / len(set(train_graph[non_edge[0]]) | set(train_graph[non_edge[1]]))
			missing_edge_score = cn_mat[u_miss, v_miss] / len(set(train_graph[missing_edge[0]]) | set(train_graph[missing_edge[1]]))

		elif index == "lhn1":
			non_edge_score = cn_mat[u_non, v_non] / len(train_graph[non_edge[0]]) * len((train_graph[non_edge[1]]))
			missing_edge_score = cn_mat[u_miss, v_miss] / len(train_graph[missing_edge[0]]) * len((train_graph[missing_edge[1]]))

		elif index == "salton":
			non_edge_score = cn_mat[u_non, v_non] / math.sqrt(len(train_graph[non_edge[0]]) * len((train_graph[non_edge[1]])))
			missing_edge_score = cn_mat[u_miss, v_miss] / math.sqrt(len(train_graph[missing_edge[0]]) * len((train_graph[missing_edge[1]])))

		elif index == "sorensen":
			non_edge_score = 2 * cn_mat[u_non, v_non] / len(train_graph[non_edge[0]]) + len((train_graph[non_edge[1]]))
			missing_edge_score = 2 * cn_mat[u_miss, v_miss] / len(train_graph[missing_edge[0]]) + len((train_graph[missing_edge[1]]))

		elif index == "hpi":
			non_edge_score = cn_mat[u_non, v_non] / min(len(train_graph[non_edge[0]]), len((train_graph[non_edge[1]])))
			missing_edge_score = cn_mat[u_miss, v_miss] / min(len(train_graph[missing_edge[0]]), len((train_graph[missing_edge[1]])))

		else:
			raise ParameterError("{} is not a valid index for funciton_name".format(index))

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5
		test_edges.remove(missing_edge)

	return total / float(num_trials)

def pa_AUC_score(G, train_graph, test_edges, num_trials):
	non_edges = n_random_non_edges(G, num_trials)

	total = 0

	for non_edge in non_edges:
		missing_edge = random.sample(test_edges, 1)[0]
		non_edge_score = len(train_graph[non_edge[0]]) * len(train_graph[non_edge[1]])
		missing_edge_score = len(train_graph[missing_edge[0]]) * len(train_graph[missing_edge[1]])

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5
		test_edges.remove(missing_edge)

	return total / float(num_trials)

def aa_ra_AUC_score(G, train_graph, test_edges, num_trials, index):
	non_edges = n_random_non_edges(G, num_trials)

	total = 0

	for non_edge in non_edges:
		missing_edge = random.sample(test_edges, 1)[0]

		if index == "aa":
			non_edge_score = sum([1 / math.log(len(train_graph[n])) for n in nx.common_neighbors(train_graph, non_edge[0], non_edge[1])])
			missing_edge_score = sum([1 / math.log(len(train_graph[n])) for n in nx.common_neighbors(train_graph, missing_edge[0], missing_edge[1])])

		elif index == "ra":
			non_edge_score = sum([1 / len(train_graph[n]) for n in nx.common_neighbors(train_graph, non_edge[0], non_edge[1])])
			missing_edge_score = sum([1 / len(train_graph[n]) for n in nx.common_neighbors(train_graph, missing_edge[0], missing_edge[1])])

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5
		test_edges.remove(missing_edge)

	return total / float(num_trials)

def predict_edges(G, train_graph, test_edges, method, num_trials, parameter = None):
	if len(test_edges) < num_trials:
		raise ParameterError("Number of test edges ({}) must at least equal the number of trials ({}) for predict_edges()".format(len(test_edges), num_trials))

	mat_score_methods = ["cn", "lp"]
	extra_mat_score_methods = ["jaccard", "lhn1", "salton", "sorensen", "hpi"]
	
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

			return mat_AUC_score(score_mat, test_edges, n_random_non_edges(G, num_trials), nodelist)

		elif method in extra_mat_score_methods:
			return extra_mat_AUC_score(G, cn_mat, train_graph, test_edges, nodelist, num_trials, method)

	elif method == "pa":
		return pa_AUC_score(G, train_graph, test_edges, num_trials)

	elif method == "aa" or method == "ra":
		return aa_ra_AUC_score(G, train_graph, test_edges, num_trials, method)
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

if __name__ == "__main__":
	G = load_graph("netscience")

	repeats = 5
	total = 0
	for i in range(repeats):
		score = k_fold_train_and_test(G.copy(), method = "sorensen", num_trials = 100, parameter = 0.04)
		print("Average AUC: {:.5f}".format(score))
		total += score
	print(total / repeats)