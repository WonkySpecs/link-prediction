import random
import math

import networkx as nx
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
				non_edge_denom = 0.5 * (len(train_graph[non_edge[0]]) + len((train_graph[non_edge[1]])))
				missing_edge_denom = 0.5 * (len(train_graph[missing_edge[0]]) + len((train_graph[missing_edge[1]])))
				
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

		#Resource Allocation extended
		#Similarity score between 2 nodes is RA + a small contribution from nodes on length 3 paths between the endpoints
		elif index == "ra_e":
			non_edge_cn = nx.common_neighbors(train_graph, non_edge[0], non_edge[1])
			path_3_nodes = set()

			#Get all nodes that are a neighbour of exactly 1 end point
			non_edge_other_neighbours_0 = set(train_graph[non_edge[0]]) - set(non_edge_cn)
			non_edge_other_neighbours_1 = set(train_graph[non_edge[1]]) - set(non_edge_cn)
			
			#Find all nodes on length 3 paths between the endpoints
			for neighbour in non_edge_other_neighbours_0:
				#If these nodes have neighbours that are neighbours of the other endpoint, they are on a path of length 3
				if set(train_graph[neighbour]) & (non_edge_other_neighbours_1 | set(non_edge_cn)):
					path_3_nodes.add(neighbour)

			for neighbour in non_edge_other_neighbours_1:
				if set(train_graph[neighbour]) & (non_edge_other_neighbours_0 | set(non_edge_cn)):
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
			missing_edge_other_neighbours_0 = set(train_graph[missing_edge[0]]) - set(missing_edge_cn)
			missing_edge_other_neighbours_1 = set(train_graph[missing_edge[1]]) - set(missing_edge_cn)
			
			for neighbour in missing_edge_other_neighbours_0:
				#If these nodes have neighbours that are neighbours of the other endpoint, they are on a path of length 3
				if set(train_graph[neighbour]) & (missing_edge_other_neighbours_1 | set(missing_edge_cn)):
					path_3_nodes.add(neighbour)

			for neighbour in missing_edge_other_neighbours_1:
				if set(train_graph[neighbour]) & (missing_edge_other_neighbours_0 | set(missing_edge_cn)):
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

		#Very similar to ra_e but takes into account the number of paths each node is on
		elif index == "ra_e2":
			non_edge_cn = nx.common_neighbors(train_graph, non_edge[0], non_edge[1])

			#Get all nodes that are a neighbour of exactly 1 end point
			non_edge_other_neighbours_0 = set(train_graph[non_edge[0]]) - set(non_edge_cn)
			non_edge_other_neighbours_1 = set(train_graph[non_edge[1]]) - set(non_edge_cn)
			non_edge_score = 0
			
			try:
				non_edge_score = sum([1 / len(train_graph[n]) for n in non_edge_cn])
			except ZeroDivisionError:
				pass

			for neighbour in non_edge_other_neighbours_0:
				#If these nodes have neighbours that are neighbours of the other endpoint, they are on a path of length 3

				try:
					non_edge_score += (parameter * len(set(train_graph[neighbour]) & (non_edge_other_neighbours_1 | set(non_edge_cn)))) / len(train_graph[neighbour])
				except ZeroDivisionError:
					pass

			for neighbour in non_edge_other_neighbours_1:
				try:
					non_edge_score += (parameter * len(set(train_graph[neighbour]) & (non_edge_other_neighbours_0 | set(non_edge_cn)))) / len(train_graph[neighbour])
				except ZeroDivisionError:
					pass

			#Repeat for missing edge
			missing_edge_cn = nx.common_neighbors(train_graph, missing_edge[0], missing_edge[1])

			#Get all nodes that are a neighbour of exactly 1 end point
			missing_edge_other_neighbours_0 = set(train_graph[missing_edge[0]]) - set(missing_edge_cn)
			missing_edge_other_neighbours_1 = set(train_graph[missing_edge[1]]) - set(missing_edge_cn)
			missing_edge_score = 0
			
			try:
				missing_edge_score = sum([1 / len(train_graph[n]) for n in missing_edge_cn])
			except ZeroDivisionError:
				pass

			for neighbour in missing_edge_other_neighbours_0:
				#If these nodes have neighbours that are neighbours of the other endpoint, they are on a path of length 3
				try:
					missing_edge_score += (parameter * len(set(train_graph[neighbour]) & (missing_edge_other_neighbours_1 | set(missing_edge_cn)))) / len(train_graph[neighbour])
				except ZeroDivisionError:
					pass

			for neighbour in missing_edge_other_neighbours_1:
				try:
					missing_edge_score += (parameter * len(set(train_graph[neighbour]) & (missing_edge_other_neighbours_0 | set(missing_edge_cn)))) / len(train_graph[neighbour])
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
	score_mat = np.eye((transition_matrix.shape[0]))

	max_diff = 1
	count = 0
	print(train_graph[train_graph.nodes()[750]])
	print(train_graph['92'])
	print(train_graph['639'])
	while max_diff > 0.01:
		old_mat = score_mat
		score_mat = np.dot(transition_matrix, score_mat)
		diff_mat = abs(old_mat - score_mat)
		max_diff = np.amax(diff_mat)
		i, j = np.unravel_index(diff_mat.argmax(), diff_mat.shape)
		print(score_mat[i, j])
		count += 1

	nodelist = list(train_graph.nodes())
	for non_edge, missing_edge in zip(non_edges, test_edges):
		u_non = nodelist.index(non_edge[0])
		v_non = nodelist.index(non_edge[1])

		u_miss = nodelist.index(missing_edge[0])
		v_miss = nodelist.index(missing_edge[1])

		s_non = score_mat[u_non, v_non] + score_mat[v_non, u_non]
		s_miss = score_mat[u_miss, v_miss] + score_mat[v_miss, u_miss]

		if s_miss > s_non:
			total += 1
		elif s_miss == s_non:
			total += 0.5

	if index == "rw":
		pass

	elif index == "rwr":
		pass
	
	return total / float(len(non_edges))