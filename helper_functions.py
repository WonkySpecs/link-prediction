import os
import random
import math
import networkx as nx
import time

class ParameterError(Exception):
	pass

def load_graph(graph_name):
	datapath = os.path.join(os.pardir, "data")
	gml_graph_dict = {"netscience" : os.path.join("netscience", "netscience.gml")}
	edgelist_graph_dict = { "condmat" 	: "CA-CondMat.txt",
							"google"	: "web-Google.txt"}

	if graph_name in gml_graph_dict:
		return nx.read_gml(os.path.join(datapath, gml_graph_dict[graph_name]))

	elif graph_name in edgelist_graph_dict:
		return nx.read_edgelist(os.path.join(datapath, edgelist_graph_dict[graph_name]))

	else:
		raise KeyError("Invalid graph_name \"{}\" passed to load_graph()".format(graph_name))

def k_set_edge_split(G, k):
	edgelist = list(G.edges())
	num_edges = len(edgelist)
	size = math.floor(float(num_edges) / k)

	subsets = []
	random.shuffle(edgelist)
	for i in range(k):
		subsets.append([e for e in edgelist[i * size: (i + 1) * size]])

	return subsets

#Networkx's non_edges function is a generator and is therefore
#not very useful for measuring AUC (can't get random non edges)
#n_random_non_edges is designed to give a random sample of n edges
# which can be used for measuring AUC.
#Beecause most nodes in the graphsbeing used are not neighbours,
#random pairs of nodes are likely to be non edges, so this 
#implementation is fast (almost no potential edgse get rejected)
def n_random_non_edges(G, n):
	non_edges = []

	while len(non_edges) < n:
		u_list = random.choices(G.nodes(), k = n - len(non_edges))
		v_list = random.choices(G.nodes(), k = n - len(non_edges))

		for (u, v) in zip(u_list, v_list):
			if v not in G[u]:
				non_edges.append((u,v))
	return non_edges