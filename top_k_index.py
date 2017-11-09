import networkx as nx
import helper_functions
import math
import time
from collections import deque

def query(index, s, t, k):
	distance_labels, loop_labels = index

	s, t = min(s, t), max(s, t)

	path_lengths = []

	for (v, d) in distance_labels[s]:
		if v == t:
			for i in range(k):
				path_lengths.append(d + loop_labels[s][i])
			#Have to skip first iteration as this is identical path to first iteration of previous loop ()the direct path between s and t)
			for i in range(1, k):
				path_lengths.append(d + loop_labels[t][i])

		else:
			for (v2, d2) in distance_labels[t]:
				if v2 == v:
					base_path_length = d + d2

					for i in range(k):
						path_lengths.append(base_path_length + loop_labels[v][i])
	path_lengths.sort()
	if len(path_lengths) < k:
		path_lengths = path_lengths + [math.inf for i in range(k - len(path_lengths))]
	return path_lengths[:k]


def v_loop_label(G, v, k):
	d_list = []
	queue = deque([(v, 0)])
	v_visits = 0

	while v_visits < k:
		try:
			cur_node, d = queue.popleft()	
		except IndexError:
			for i in range(k - v_visits):
				d_list.append(math.inf)
			break

		if cur_node == v:
			v_visits += 1
			d_list.append(d)
		for n in nx.neighbors(G, cur_node):
			if n >= v:
				queue.append((n, d + 1))

	return d_list

def get_loop_labels(G, k):
	loop_labels = []
	for node in G.nodes():
		loop_labels.append(v_loop_label(G, node, k))

	return loop_labels

def v_distance_label(G, v, k, index):
	distance_labels, loop_labels = index
	queue = deque([(v, 0)])
	while queue:
		u, delta = queue.popleft()
		if delta < max(query(index, v, u, k)):
			distance_labels[u].append((v, delta))
			for w in nx.neighbors(G, u):
				if w > v:
					queue.append((w, delta + 1))
	return distance_labels

def get_distance_labels(G, k, loop_labels):
	distance_labels = [[] for i in range(len(G.nodes()))]
	for node in G.nodes():
		#print(distance_labels)
		distance_labels = v_distance_label(G, node, k, (distance_labels, loop_labels))

	return distance_labels

def construct_index(G, k):
	ll = get_loop_labels(G, k)
	dl = get_distance_labels(G, k, ll)

	return (dl, ll)

#Compute vertex labelling for graph - for now just returns nodes in order
def vertex_ordering(G):
	count = 0
	vo = dict()
	for node in G.nodes():
		vo[node] = count
		count += 1
	return vo

def write_index_to_file(filename, index):
	distance_labels, loop_labels = index
	with open(filename, "w") as file:
		file.write("loop_labels\n")
		for loop_label in loop_labels:
			s = ""
			for v in loop_label:
				s += str(v) + ","
			s = s[:-1] + "\n"	
			file.write(s)
		file.write("distance_labels\n")
		for distance_label in distance_labels:
			s = ""
			for (v, d) in distance_label:
				s += "({},{}),".format(v, d)
			s = s[:-1] + "\n"
			file.write(s)

def read_index_from_file(filename):
	loop_labels_done = False
	loop_labels = []
	distance_labels = []
	with open(filename, "r") as file:
		for line in file:
			if line.endswith("\n"):
				line = line[:-1]
			if line == "loop_labels":
				continue
			if line == "distance_labels":
				loop_labels_done = True
				continue

			if not loop_labels_done:
				loop_label = []
				for d in line.split(","):
					if d == "inf":
						loop_label.append(math.inf)
					else:
						loop_label.append(int(d))
				loop_labels.append(loop_label)

			else:
				distance_label = []
				pairs = line.replace(")", "").split("(")[1:]
				for pair in pairs:
					pair = pair.split(",")
					distance_label.append((int(pair[0]), int(pair[1])))
				distance_labels.append(distance_label)
	return distance_labels, loop_labels

if __name__ == "__main__":
	G = nx.Graph()
	G.add_edge(0, 1)
	G.add_edge(0, 2)
	G.add_edge(1, 2)
	G.add_edge(1, 3)
	G.add_edge(1, 4)
	G.add_edge(2, 4)
	G.add_edge(4, 5)

	G = helper_functions.load_graph("netscience")
	vertex_id_map = vertex_ordering(G)

	ordered_G = nx.Graph()
	for original_node, new_id in vertex_id_map.items():
		ordered_G.add_node(new_id)
		for neighbor in nx.neighbors(G, original_node):
			ordered_G.add_edge(new_id, vertex_id_map[neighbor])
	print(len(ordered_G.nodes()))

	#ordered_G = nx.path_graph(6)
	
	k = 6
	index = construct_index(ordered_G, k)

	write_index_to_file("test.idx", index)
	index = read_index_from_file("test.idx")

	print(query(index, 32, 33, k))

	# for i in range(len(G.nodes())):
	# 	for j in range(i + 1, len(G.nodes())):
	# 		print("Top 4 lengths between {} {}:".format(i, j))
	# 		print(query(index, i, j, 4))
	# 		print("-----------------")



#G = helper_functions.load_graph("netscience")
# G = nx.path_graph(5)
# print(G.edges())
# vertex_id_map = vertex_ordering(G)

# ordered_G = nx.Graph()
# for original_node, new_id in vertex_id_map.items():
# 	ordered_G.add_node(new_id)
# 	for neighbor in nx.neighbors(G, original_node):
# 		ordered_G.add_edge(new_id, vertex_id_map[neighbor])

# start = time.clock()
# get_loop_labels(ordered_G, 4)
# print(time.clock() - start)
