import csv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from helper_functions import *

source_folder = os.path.join(os.pardir, "results", "random", "pa", "1")

indices = ["lp", "cn", "lhn1_e", "lhn1", "hpi_e", "hpi", "hdi_e", "hdi", "ra_e", "ra", "pa", "ra_e2"]

index_i_dict = None
graph_dict = dict()
graph_name_list = []

results_file = None
for f in os.listdir(source_folder):
	if f.endswith(".csv"):
		if results_file:
			print("More than one csv file found in {}, sort that crap out".format(source_folder))
			exit()
		else:
			results_file = os.path.join(source_folder, f)

#Get results from file
with open(results_file) as csvfile:
	reader = csv.reader(csvfile, delimiter = ",")
	for row in reader:
		if index_i_dict:
			graph_name = row[0]
			graph_name_list.append(graph_name)
			graph_dict[graph_name] = dict()
			graph_dict[graph_name]["AUC"] = dict()

			for index, i in index_i_dict.items():
				graph_dict[graph_name]["AUC"][index] = row[i]
		else:
			index_i_dict = {index : row.index(index) for index in indices}

#Calculate relevant graph properties
for graph_name in graph_name_list:
	total_nodes = 0
	max_deg = 0
	max_deg_node = None

	G = load_graph(graph_name, datapath = source_folder)

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

plot_vals = {i : dict() for i in indices}
for index in indices:
	for graph in graph_dict.values():
		d = graph["avg_degree"]
		v = float(graph["AUC"][index])

		if d in plot_vals[index].keys():
			plot_vals[index][d].append(v)
		else:
			plot_vals[index][d] = [v]

list_avg = lambda l : sum(l) / len(l)

for (local_index, extended_index) in zip(["lhn1", "hpi", "hdi"], ["lhn1_e", "hpi_e", "hdi_e"]):
	plt.plot(list(plot_vals["cn"].keys()), [list_avg(l) for l in plot_vals["cn"].values()], linestyle = '--', marker = 'o', color = (0.55, 0.55, 0.55))
	plt.plot(list(plot_vals["lp"].keys()), [list_avg(l) for l in plot_vals["lp"].values()], linestyle = '--', marker = 'x', color = (0.45, 0.45, 0.45))
	plt.plot(list(plot_vals[local_index].keys()), [list_avg(l) for l in plot_vals[local_index].values()], linestyle = '-', marker = 'o', color = (0, 0.6, 0))
	plt.plot(list(plot_vals[extended_index].keys()), [list_avg(l) for l in plot_vals[extended_index].values()], linestyle = '-', marker = 'x', color = (0, 0.8, 0))
	plt.xlabel('Average degree')
	plt.ylabel('AUC')
	plt.title("Performance of {} and {}".format(local_index.upper(), extended_index.upper()))
	plt.savefig("test_" + local_index)
	plt.clf()

plt.plot(list(plot_vals["cn"].keys()), [list_avg(l) for l in plot_vals["cn"].values()], linestyle = '--', marker = 'o', color = (0.55, 0.55, 0.55))
plt.plot(list(plot_vals["lp"].keys()), [list_avg(l) for l in plot_vals["lp"].values()], linestyle = '--', marker = 'x', color = (0.45, 0.45, 0.45))
plt.plot(list(plot_vals["ra"].keys()), [list_avg(l) for l in plot_vals["ra"].values()], linestyle = '-', marker = 'o', color = (0, 0.5, 0))
plt.plot(list(plot_vals["ra_e"].keys()), [list_avg(l) for l in plot_vals["ra_e"].values()], linestyle = '-', marker = 'x', color = (0, 0.75, 0))
plt.plot(list(plot_vals["ra_e2"].keys()), [list_avg(l) for l in plot_vals["ra_e2"].values()], linestyle = '-', marker = '^', color = (0, 0.95, 0))
plt.xlabel('Average degree')
plt.ylabel('AUC')
plt.title("Performance of {}, {}, and {}".format("RA", "RA_E", "RA_E2"))
plt.savefig("test_ra")
exit()


#Display graphs in 2 row grid
gs = gridspec.GridSpec(2, (len(indices) + 1) // 2)
gs.update(wspace = 0.4)
fig = plt.figure()
for n, index in enumerate(indices):
	ax = plt.subplot(gs[(n + 1) % 2, n // 2])
	ax.scatter([graph_dict[graph_name]["avg_degree"] for graph_name in graph_name_list], [graph_dict[graph_name][index] for graph_name in graph_name_list])
	ax.set_title(index)
	if n == 1:
		ax.set_xlabel("Average degree")
		ax.set_ylabel("AUC")
plt.subplots_adjust(hspace = 0.3, wspace = 0.15)
fig.suptitle("Index performance on random graphs generated using a preferential attachment model", size = 12)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.savefig("ASDF")
plt.show()