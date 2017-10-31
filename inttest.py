import networkx as nx
import os

print("file\t\tn\tm")
for f in os.listdir(os.path.join(os.curdir, "data", "rocketfuel_maps_cch.tar")):
	if f.endswith(".el"):
		G = nx.read_edgelist(os.path.join(os.curdir, "data", "rocketfuel_maps_cch.tar", f))
		print("{:12}\t{}\t{}".format(f, len(G.nodes()), len(G.edges())))