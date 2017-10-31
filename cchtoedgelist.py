import sys, os

datapath = os.path.join(os.pardir, "data", "rocketfuel_maps_cch.tar")
#check filename is ok
def convert(filename):
	with open(filename) as cch_f:
		out_filename = "{}.el".format(filename[:-4])
		edgelist_f = open(out_filename, "w")
		edgelist_f.write("# Edgelist converted from cch file {}\n".format(filename))
		for line in cch_f:
			parts = line.split()
			u = parts[0]
			endpoints = []
			for part in parts:
				if part.startswith("<") and part.endswith(">"):
					endpoints.append(part[1:-1])
			for v in endpoints:
				edgelist_f.write("{}\t{}\n".format(u, v))

if len(sys.argv) == 2:
	filename = os.path.join(os.pardir, "data", "rocketfuel_maps_cch.tar", sys.argv[1])
elif len(sys.argv) > 2:
	pass
else:
	#default file for testing
	filename = os.path.join(datapath, "7018.cch")
	for file in os.listdir(datapath):
		if file.endswith(".cch") and not file.startswith("README") and not file.endswith(".pop.cch"):
			convert(os.path.join(datapath, file))