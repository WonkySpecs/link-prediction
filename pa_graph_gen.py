#Generate graph
#Calc AUC with measures - all or just interesting?
#Save graph and results (edgelist/csv)

from main import measure_index_performance
import networkx as nx
import csv
import os
import time
from decimal import *

num_graphs = 3
graph_n = 1000
graph_m_list = [(1 + 3 * i) for i in range(11)]
test_indices = ["cn", "lhn1", "hpi", "hdi", "lhn1_e", "hpi_e", "hdi_e", "ra", "ra_e", "ra_e2", "pa"]

with open("pa-results.csv", 'w', newline = '') as csvfile:
	writer = csv.writer(csvfile, delimiter = ",")
	writer.writerow(["graph_name"] + test_indices)

	for m in graph_m_list:
		print("m = " + str(m))
		for graph in range(num_graphs):
			print(graph + 1)
			graph_name = str(graph_n) + "-" + str(m) + "pa" + str(hash(time.time()))
			G = nx.barabasi_albert_graph(graph_n, m)
			AUC_list = []

			for index in test_indices:
				AUC_list.append(round(Decimal(measure_index_performance(index, G)), 8))
			writer.writerow([graph_name] + AUC_list)

			with open(os.path.join(os.pardir, "data", "random", "pa", graph_name), mode = "wb") as out_file:
				nx.write_edgelist(G, out_file, data = False)
