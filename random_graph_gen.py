#Generate graph
#Calc AUC with measures - all or just interesting?
#Save graph and results (edgelist/csv)

from main import measure_index_performance
import networkx as nx
import csv
import os
import time
from decimal import *

#Accepted algorithms are "pa" (preferential attachment) and "sw" (small world)"
algorithm = "sw"
num_graphs = 3
n = 1000
test_indices = ["cn", "lp", "lhn1", "hpi", "hdi", "lhn1_e", "hpi_e", "hdi_e", "ra", "ra_e", "ra_e2", "pa"]

m_list = [(1 + 3 * i) for i in range(11)]
k_list = [2 * i for i in range(1, 8)]

if algorithm == "pa":
	minor_param_list = m_list
elif algorithm == "sw":
	minor_param_list = k_list

random_results_folder = os.path.join(os.pardir, "results", "random", algorithm)
max_folder_num = 0
for folder in os.listdir(random_results_folder):
	if int(folder) > max_folder_num:
		max_folder_num = int(folder)

output_folder = os.path.join(random_results_folder, str(max_folder_num + 1))
os.mkdir(output_folder)

result_file = os.path.join(output_folder, "results.csv")
with open(result_file, 'w', newline = '') as csvfile:
	writer = csv.writer(csvfile, delimiter = ",")
	writer.writerow(["graph_name"] + test_indices)

	for p2 in minor_param_list:
		for graph in range(num_graphs):
			print(graph + 1)
			graph_name = str(n) + "-" + str(p2) + algorithm + str(hash(time.time()))
			if algorithm == "pa":
				G = nx.barabasi_albert_graph(n, p2)
			elif algorithm == "sw":
				G = nx.watts_strogatz_graph(n, p2, 0.1)
				G.remove_nodes_from(nx.isolates(G))

			AUC_list = []

			for index in test_indices:
				AUC_list.append(round(Decimal(measure_index_performance(index, G)), 8))
			writer.writerow([graph_name] + AUC_list)

			with open(os.path.join(output_folder, graph_name), mode = "wb") as graph_out_file:
				nx.write_edgelist(G, graph_out_file, data = False)
