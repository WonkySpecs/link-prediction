import csv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from helper_functions import *

source_folder = os.path.join(os.pardir, "results")

indices = ["cn", "lhn1", "hpi", "hdi", "lhn1_e", "hpi_e", "hdi_e", "ra", "ra_e", "ra_e2", "pa"]

index_i_dict = None
graph_dict = dict()
graph_name_list = []

results_file = None
for f in source_folder:
	if f.endswith(".csv"):
		if results_file:
			print("More than one csv file found in {}, sort that crap out".format(source_folder))
			exit()
		else:
			results_file = os.path.join(source_folder, f)

#Get results from file
with open(results_file, newline = '') as csvfile:
	reader = csv.reader(csvfile, delimiter = ",")
	for row in reader:
		if index_i_dict:
			graph_name = row[0]
			graph_name_list.append(graph_name)
			G = graphs[graph_name]
			graph_dict[graph_name] = dict()

			for index, i in index_i_dict.items():
				graph_dict[graph_name][index] = row[i]
		else:
			index_i_dict = {index : row.index(index) for index in indices}

#Calculate relevant graph properties
for graph_name in graph_name_list:
	total_nodes = 0
	max_deg = 0
	max_deg_node = None

	G = load_graph(graph_name, foler = source_folder)

	for node in G.nodes():
		n = len(G[node])
		total_nodes += n
		if n > max_deg:
			max_deg = n
			max_deg_node = node

	avg_deg = total_nodes / G.number_of_nodes()
	graph_dict[graph_name]["avg_degree"] = avg_deg
	graph_dict[graph_name]["max_degree"] = max_deg
	graph_dict[graph_name]["avg_clustering"] = nx.average_clustering(G)

#Display graphs in 2 row grid
gs = gridspec.GridSpec(2, (len(indices) + 1) // 2)
gs.update(wspace = 0.4)

for n, index in enumerate(indices):
	ax = plt.subplot(gs[(n + 1) % 2, n // 2])
	ax.scatter([graph_dict[graph_name]["avg_degree"] for graph_name in graph_name_list], [graph_dict[graph_name][index] for graph_name in graph_name_list])
	ax.set_title(index)

plt.show()