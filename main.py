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
	for i in range(len(non_edges)):
		missing_edge = test_edges[i]
		non_edge = non_edges[i]

		non_edge_score = score_mat[nodelist.index(non_edge[0]), nodelist.index(non_edge[1])]
		missing_edge_score = score_mat[nodelist.index(missing_edge[0]), nodelist.index(missing_edge[1])]

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5

	return total / float(len(non_edges))

#These indices require more processing than just looking up matrix elements
def extra_mat_AUC_score(cn_mat, train_graph, test_edges, nodelist, non_edges, index):
	total = 0

	for non_edge, missing_edge in zip(non_edges, test_edges):
		u_non = nodelist.index(non_edge[0])
		v_non = nodelist.index(non_edge[1])

		u_miss = nodelist.index(missing_edge[0])
		v_miss = nodelist.index(missing_edge[1])

		with np.errstate(all = "raise"):
			if index == "jaccard":
				non_edge_denom = len(set(train_graph[non_edge[0]]) | set(train_graph[non_edge[1]]))
				missing_edge_denom = len(set(train_graph[missing_edge[0]]) | set(train_graph[missing_edge[1]]))

			elif index == "lhn1":
				non_edge_denom = len(train_graph[non_edge[0]]) * len((train_graph[non_edge[1]]))
				missing_edge_denom = len(train_graph[missing_edge[0]]) * len((train_graph[missing_edge[1]]))

			elif index == "salton":
				non_edge_denom = math.sqrt(len(train_graph[non_edge[0]]) * len((train_graph[non_edge[1]])))
				missing_edge_denom = math.sqrt(len(train_graph[missing_edge[0]]) * len((train_graph[missing_edge[1]])))

			elif index == "sorensen":
				non_edge_score = 0.5 * (len(train_graph[non_edge[0]]) + len((train_graph[non_edge[1]])))
				missing_edge_score = 0.5 * (len(train_graph[missing_edge[0]]) + len((train_graph[missing_edge[1]])))
				
			elif index == "hpi":
				non_edge_denom = min(len(train_graph[non_edge[0]]), len((train_graph[non_edge[1]])))
				missing_edge_denom = min(len(train_graph[missing_edge[0]]), len((train_graph[missing_edge[1]])))
				
			elif index == "hdi":
				non_edge_denom = max(len(train_graph[non_edge[0]]), len((train_graph[non_edge[1]])))
				missing_edge_denom = max(len(train_graph[missing_edge[0]]), len((train_graph[missing_edge[1]])))

			else:
				raise ParameterError("{} is not a valid index for extra_mat_AUC_score()".format(index))

			if non_edge_denom > 0:
				non_edge_score = cn_mat[u_non, v_non] / non_edge_denom
			else:
				non_edge_score = 0

			if missing_edge_denom > 0:
				missing_edge_score = cn_mat[u_miss, v_miss] / missing_edge_denom
			else:
				missing_edge_score = 0

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5

	return total / float(len(non_edges))

def pa_AUC_score(train_graph, test_edges, non_edges):
	total = 0

	for non_edge, missing_edge in zip(non_edges, test_edges):
		missing_edge = random.sample(test_edges, 1)[0]
		non_edge_score = len(train_graph[non_edge[0]]) * len(train_graph[non_edge[1]])
		missing_edge_score = len(train_graph[missing_edge[0]]) * len(train_graph[missing_edge[1]])

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5

	return total / float(len(non_edges))

def aa_ra_AUC_score(train_graph, test_edges, non_edges, index, parameter = None):
	total = 0

	for non_edge, missing_edge in zip(non_edges, test_edges):
		if index == "aa":
			try:
				non_edge_score = sum([1 / math.log(len(train_graph[n])) for n in nx.common_neighbors(train_graph, non_edge[0], non_edge[1])])
			except ZeroDivisionError:
				non_edge_score = 0

			try:
				missing_edge_score = sum([1 / math.log(len(train_graph[n])) for n in nx.common_neighbors(train_graph, missing_edge[0], missing_edge[1])])
			except ZeroDivisionError:
				missing_edge_score = 0
			

		elif index == "ra":
			try:
				non_edge_score = sum([1 / len(train_graph[n]) for n in nx.common_neighbors(train_graph, non_edge[0], non_edge[1])])
			except ZeroDivisionError:
				non_edge_score = 0

			try:
				missing_edge_score = sum([1 / len(train_graph[n]) for n in nx.common_neighbors(train_graph, missing_edge[0], missing_edge[1])])
			except ZeroDivisionError:
				missing_edge_score = 0

		elif index == "ra_e":
			non_edge_cn = nx.common_neighbors(train_graph, non_edge[0], non_edge[1])
			path_3_nodes = set()

			#Get all nodes that are a neighbour of exactly 1 end point
			non_edge_other_neighbours_0 = set(G[non_edge[0]]) - set(non_edge_cn)
			non_edge_other_neighbours_1 = set(G[non_edge[1]]) - set(non_edge_cn)
			
			for neighbour in non_edge_other_neighbours_0:
				#If these nodes have neighbours that are neighbours of the other endpoint, they are on a path of length 3
				if set(G[neighbour]) & (non_edge_other_neighbours_1 | set(non_edge_cn)):
					path_3_nodes.add(neighbour)

			for neighbour in non_edge_other_neighbours_1:
				if set(G[neighbour]) & (non_edge_other_neighbours_0 | set(non_edge_cn)):
					path_3_nodes.add(neighbour)

			non_edge_score = 0
			try:
				non_edge_score = sum([1 / len(train_graph[n]) for n in non_edge_cn])
			except ZeroDivisionError:
				pass

			try:
				non_edge_score += parameter * sum([1 / len(train_graph[n]) for n in path_3_nodes])
			except ZeroDivisionError:
				pass

			#Repeat for missing edge

			missing_edge_cn = nx.common_neighbors(train_graph, missing_edge[0], missing_edge[1])
			path_3_nodes = set()

			#Get all nodes that are a neighbour of exactly 1 end point
			missing_edge_other_neighbours_0 = set(G[missing_edge[0]]) - set(missing_edge_cn)
			missing_edge_other_neighbours_1 = set(G[missing_edge[1]]) - set(missing_edge_cn)
			
			for neighbour in missing_edge_other_neighbours_0:
				#If these nodes have neighbours that are neighbours of the other endpoint, they are on a path of length 3
				if set(G[neighbour]) & (missing_edge_other_neighbours_1 | set(missing_edge_cn)):
					path_3_nodes.add(neighbour)

			for neighbour in missing_edge_other_neighbours_1:
				if set(G[neighbour]) & (missing_edge_other_neighbours_0 | set(missing_edge_cn)):
					path_3_nodes.add(neighbour)

			missing_edge_score = 0
			try:
				missing_edge_score = sum([1 / len(train_graph[n]) for n in missing_edge_cn])
			except ZeroDivisionError:
				pass

			try:
				missing_edge_score += parameter * sum([1 / len(train_graph[n]) for n in path_3_nodes])
			except ZeroDivisionError:
				pass

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5

	return total / float(len(non_edges))

def experimental_AUC_score(train_graph, test_edges, nodelist, lp_mat, non_edges, index):
	total = 0
	for non_edge, missing_edge in zip(non_edges, test_edges):
		u_non = nodelist.index(non_edge[0])
		v_non = nodelist.index(non_edge[1])

		u_miss = nodelist.index(missing_edge[0])
		v_miss = nodelist.index(missing_edge[1])
		
		with np.errstate(all = "raise"):
			if index == "lhn1_e":
				non_edge_denom = len(train_graph[non_edge[0]]) * len((train_graph[non_edge[1]]))
				missing_edge_denom = len(train_graph[missing_edge[0]]) * len((train_graph[missing_edge[1]]))

			elif index == "salton_e":
				non_edge_denom = math.sqrt(len(train_graph[non_edge[0]]) * len((train_graph[non_edge[1]])))
				missing_edge_denom = math.sqrt(len(train_graph[missing_edge[0]]) * len((train_graph[missing_edge[1]])))

			elif index == "hpi_e":
				non_edge_denom = min(len(train_graph[non_edge[0]]), len((train_graph[non_edge[1]])))
				missing_edge_denom = min(len(train_graph[missing_edge[0]]), len((train_graph[missing_edge[1]])))
				
			elif index == "hdi_e":
				non_edge_denom = max(len(train_graph[non_edge[0]]), len((train_graph[non_edge[1]])))
				missing_edge_denom = max(len(train_graph[missing_edge[0]]), len((train_graph[missing_edge[1]])))

			else:
				raise ParameterError("{} is not a valid index for extra_mat_AUC_score()".format(index))

			if non_edge_denom > 0:
				non_edge_score = lp_mat[u_non, v_non] / non_edge_denom
			else:
				non_edge_score = 0

			if missing_edge_denom > 0:
				missing_edge_score = lp_mat[u_miss, v_miss] / missing_edge_denom
			else:
				missing_edge_score = 0

		if missing_edge_score > non_edge_score:
			total += 1
		elif missing_edge_score == non_edge_score:
			total += 0.5

	return total / float(len(non_edges))

def rw_AUC_score(train_graph, test_edges, non_edges, index):
	total = 0
	a_mat = nx.adjacency_matrix(train_graph)
	row_sums = a_mat.sum(axis = 1)
	#If a node has become an isolate during k-fold, row sum will be 0 which causes an division error
	with np.errstate(invalid = "ignore"):
		transition_matrix = a_mat / row_sums
	#Division errors put nan into matrix, replace nans with 0 (no chance of transition)
	transition_matrix = np.nan_to_num(transition_matrix)
	transition_matrix = np.transpose(transition_matrix)
	score_mat = []

	for node in train_graph.nodes():
		vec = np.zeros((transition_matrix.shape[0], 1))
		vec[node] = 1
		max_diff = 1

		while max_diff > 0.002:
			new_vec = np.dot(transition_matrix, vec)
			max_diff = max(abs(new_vec - vec))[0,0]
			vec = new_vec

		#print(vec)
		print("----------------------")

	if index == "rw":
		pass
	elif index == "rwr":
		pass
	
	return total / float(len(non_edges))

def predict_edges(G, train_graph, test_edges, method, num_trials, parameter = None):
	mat_score_methods = ["cn", "lp"]
	extra_mat_score_methods = ["jaccard", "lhn1", "salton", "sorensen", "hpi", "hdi"]
	sum_indices = ["aa", "ra", "ra_e"]
	random_walker_indices = ["rw", "rwr"]
	experimental_indices = ["hpi_e", "hdi_e", "salton_e", "lhn1_e", "ra_e"]

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

if __name__ == "__main__":
	repeats = 10
	for graph in ["netscience", "lastfm", "condmat", "power"]:
		print(graph)
		G = load_graph(graph)
		for method in ["jaccard", "salton", "salton_e", "lhn1", "lhn1_e", "hdi", "hdi_e", "hpi", "hpi_e", "aa", "ra", "ra_e"]:
			total = 0
			print(method)
			for i in range(repeats):
				score = k_fold_train_and_test(G.copy(), method = method, num_trials = 200, parameter = 0.02)
				total += score
			print("Average AUC: {:.4f}".format(total / repeats))
		print("------------\n")