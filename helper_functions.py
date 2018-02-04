import os
import random
import math
import networkx as nx
import time

class ParameterError(Exception):
	pass

def load_graph(graph_name):
	datapath = os.path.join(os.pardir, "data")
	gml_graph_dict = {  "netscience" : os.path.join("netscience", "netscience.gml"),
						"karate" : "karate.gml",
						"power"	 : "power.gml",
						"books"	 : "polbooks.gml" }
						
	edgelist_graph_dict = { "condmat" 	: "CA-CondMat.txt",
							"google"	: "web-Google.txt",
							"facebook"	: "facebook_combined.txt",
							"lastfm"	: os.path.join("lastfm", "user_friends.dat"),
							"movies"	: "rec-movielens-user-movies-10m.edges"}

	G = nx.Graph()

	if graph_name in gml_graph_dict:
		G = nx.read_gml(os.path.join(datapath, gml_graph_dict[graph_name]), label = 'id')

		#We use ids to override labels, might want to overwrite this behaviour at some point?
		# 	G = nx.read_gml(os.path.join(datapath, gml_graph_dict[graph_name]))

	elif graph_name in edgelist_graph_dict:
		G = nx.read_edgelist(os.path.join(datapath, edgelist_graph_dict[graph_name]))

	elif graph_name == "test":
		#This is graph 'c' pictured in paper
		G = nx.path_graph(5)
		G.add_edge(0, 5)
		G.add_edge(5, 6)
		G.add_edge(6, 7)
		G.add_edge(7, 4)

		G.add_edge(0, 8)
		G.add_edge(8, 9)
		G.add_edge(9, 10)
		G.add_edge(10, 4)

		G.add_edge(5, 2)
		G.add_edge(6, 1)
		G.add_edge(6, 3)
		G.add_edge(7, 2)

		G.add_edge(8, 2)
		G.add_edge(9, 3)
		G.add_edge(9, 1)
		G.add_edge(10, 2)
	else:
		raise KeyError("Invalid graph_name \"{}\" passed to load_graph()".format(graph_name))

	G.remove_nodes_from(nx.isolates(G))

	#Remap graph nodes into range 0 - (n - 1)
	new_old_map = {new_node : old_node for new_node, old_node in enumerate(G.nodes())}
	old_new_map = {old_node : new_node for new_node, old_node in new_old_map.items()}
	ordered_G = nx.Graph()

	for new_node in range(G.number_of_nodes()):
		ordered_G.add_node(new_node)
		for old_neighbour in G[new_old_map[new_node]]:
			ordered_G.add_edge(new_node, old_new_map[old_neighbour])
	return G

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