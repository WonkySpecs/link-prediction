import networkx as nx
import helper_functions
import math
import time
import random
from collections import deque

#Returns a list of k int distances for the shortest paths between nodes s and t
def query(index, s, t, k):
	distance_labels, loop_labels = index

	s, t = min(s, t), max(s, t)

	path_lengths = []

	for (v, d, c) in distance_labels[s]:
		#NOTE: This almost never occurs - possibly s and t should be other way around?
		if v == t:
			for (loop_d, n) in loop_labels[s]:
				for i in range(n):
					path_lengths.append(d + loop_d)
			for (loop_d, n) in loop_labels[t]:
				#Skip the 0 loop on t, as this is the same path as using the 0 loop for s (direct path)
				if loop_d == 0:
					n -= 1
				for i in range(n):
					path_lengths.append(d + loop_d)
		else:
			for (v2, d2, c2) in distance_labels[t]:
				#Matching mid point vertices between s and t
				if v2 == v:
					base_path_length = d + d2
					#There are c * c2 total paths from s to t which pass through v
					for num_paths in range(c * c2):
						for(loop_d, n) in loop_labels[v]:
							path_lengths.append(base_path_length + loop_d)

	path_lengths.sort()
	if len(path_lengths) < k:
		path_lengths = path_lengths + [math.inf for i in range(k - len(path_lengths))]
	return path_lengths[:k]

def v_loop_label(G, v, k):
	d_list = []
	d_counts = dict()
	queue = deque([(v, 0)])
	v_visits = 0

	#BFS to find k shortest cycles from v to v
	while v_visits < k:
		try:
			cur_node, d = queue.popleft()	
		except IndexError:
			break

		#Reached v again
		if cur_node == v:
			v_visits += 1
			try:
				d_counts[d] += 1
			except KeyError:
				d_counts[d] = 1

		for n in nx.neighbors(G, cur_node):
			if n >= v:
				queue.append((n, d + 1))

	#At this point, d_counts is a dict of path lengths:occurences
	#Want to convert this to a list of tuples containing the k shortest paths
	total_loops = 0

	while total_loops < k and d_counts:
		min_d = min(d_counts.keys())
		total_loops += d_counts[min_d]

		if total_loops <= k:
			d_list.append((min_d, d_counts[min_d]))
		else:
			d_list.append((min_d, d_counts[min_d] - (total_loops - k)))
		del d_counts[min_d]

	#Not enough paths, fill with inf
	#This occurs when a node has no larger neighbours
	if total_loops < k:
		d_list.append((math.inf, k - total_loops))

	return d_list

def get_loop_labels(G, k):
	loop_labels = []
	for n in range(len(G.nodes())):
		loop_labels.append(v_loop_label(G, n, k))

	return loop_labels

def v_distance_label(G, v, k, index):
	distance_labels, loop_labels = index
	queue = deque([(v, 0)])
	counts = {(v, 0) : 1}
	count = 0
	while queue:
		u, delta = queue.popleft()
		count += 1
		if delta < max(query(index, v, u, k)):
			distance_labels[u].append((v, delta, counts[u, delta]))
			for w in nx.neighbors(G, u):
				if w > v:
					#Only want 1 instance of (n,d) for any given n and d
					#If it already exists in the queue, just increment count for this pair
					if (w, delta + 1) in counts:
						counts[(w, delta + 1)] += 1
					else:
						queue.append((w, delta + 1))
						counts[(w, delta + 1)] = 1
	return distance_labels, count

def get_distance_labels(G, k, loop_labels):
	distance_labels = [[] for i in range(len(G.nodes()))]
	total = 0
	for n in range(len(G.nodes())):
		distance_labels, count = v_distance_label(G, n, k, (distance_labels, loop_labels))
		total += count
	print("average number of pops was {}".format(total/len(G.nodes())))

	return distance_labels

def construct_index(G, k):
	start = time.clock()
	ll = get_loop_labels(G, k)
	#print("Loop labels done in {} seconds".format(time.clock() - start))
	start = time.clock()
	dl = get_distance_labels(G, k, ll)
	#print("Distance labels done in {} seconds".format(time.clock() - start))

	return (dl, ll)

#Compresses vertices labels into the range 0...n
#where n is the number of vertices
#This name is awful, change it
def in_order_vertex(G):
	vertex_id_map = dict()
	count = 0
	for node in G.nodes():
		vertex_id_map[node] = count
		count += 1

	ordered_G = nx.Graph()
	for original_node, new_id in vertex_id_map.items():
		ordered_G.add_node(new_id)
		for neighbor in nx.neighbors(G, original_node):
			ordered_G.add_edge(new_id, vertex_id_map[neighbor])
	return ordered_G, vertex_id_map

#Sorts vertices in order of decreasing degree
#Returns a dict of original:new vertice id
def vertex_ordering(G):
	count = 0
	vo = dict()
	deg_list = [G.degree(node) for node in G.nodes()]
	deg_count_map = dict()
	for d in deg_list:
		if d in deg_count_map:
			deg_count_map[d] += 1
		else:
			deg_count_map[d] = 1
	while deg_count_map:
		largest_deg = max(deg_count_map.keys())

		next_index = -1
		while deg_count_map[largest_deg] > 0:
			next_index = deg_list.index(largest_deg, next_index + 1)
			vo[G.nodes()[next_index]] = count
			deg_count_map[largest_deg] -= 1
			count += 1

		del deg_count_map[largest_deg]

	return vo

#Builds an isomorphic graph for G with the nodes in an order more optimized for indexing
def optimize_vertex_order(G):
	vertex_id_map = vertex_ordering(G)

	ordered_G = nx.Graph()
	for original_node, new_id in vertex_id_map.items():
		ordered_G.add_node(new_id)
		for neighbor in nx.neighbors(G, original_node):
			ordered_G.add_edge(new_id, vertex_id_map[neighbor])
	return ordered_G, vertex_id_map

def write_index_to_file(filename, index, vertex_map):
	distance_labels, loop_labels = index
	with open(filename, "w") as file:
		file.write("vertex_map\n")
		for key, item in vertex_map.items():
			file.write("{}:{}\n".format(key, item))

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
			for (v, d, c) in distance_label:
				s += "({},{},{}),".format(v, d, c)
			s = s[:-1] + "\n"
			file.write(s)

def read_index_from_file(filename):
	section = "vertex_map"
	vertex_map = dict()
	loop_labels = []
	distance_labels = []
	with open(filename, "r") as file:
		for line in file:
			if line.endswith("\n"):
				line = line[:-1]

			if line == "vertex_map" or line == "loop_labels" or line == "distance_labels":
				section = line
				continue

			if section == "vertex_map":
				parts = line.split(":")
				vertex_map[int(parts[0])] = int(parts[1])

			elif section == "loop_labels":
				loop_label = []
				pairs = line.replace(")", "").split("(")[1:]

				for pair in pairs:
					pair = pair.split(",")
					if pair[0] == "inf":
						loop_label.append((float(pair[0]), int(pair[1])))
					else:
						loop_label.append((int(pair[0]), int(pair[1])))

				loop_labels.append(loop_label)

			else:
				distance_label = []
				pairs = line.replace(")", "").split("(")[1:]
				for pair in pairs:
					pair = pair.split(",")
					distance_label.append((int(pair[0]), int(pair[1]), int(pair[2])))
				distance_labels.append(distance_label)
	return distance_labels, loop_labels, vertex_map

if __name__ == "__main__":
	G = helper_functions.load_graph("test")
	orig_G = G.copy()

	indices = []
	ordered_G, vertex_map = optimize_vertex_order(G)

	for k in [2, 4, 8]:
		print("Constructing top-{} index".format(k))
		start = time.clock()
		index = construct_index(ordered_G, k)
		print("Took {} seconds".format(time.clock() - start))
		indices.append(index)
		write_index_to_file("test{}.idx".format(k), index, vertex_map)
		dl, ll, vm = read_index_from_file("test{}.idx".format(k))
		print(dl)
		print(ll)


	# print("Testing")
	# num_tests = 20
	# for test in range(num_tests):
	# 	u = random.randint(0, 33)
	# 	v = random.randint(0, 33)
	# 	qs = [query(indices[0], u, v, 2), query(indices[1], u, v, 4), query(indices[2], u, v, 8)]

	# 	if qs[0] == qs[1][:2]:
	# 		if qs[1] == qs[2][:4]:
	# 			#print(qs[2])
	# 			pass
	# 		else:
	# 			print("top 4 != top 8")
	# 	else:
	# 		print("top 2 != top 4")
	# print("done")