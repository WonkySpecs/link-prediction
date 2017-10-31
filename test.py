import networkx as nx
import main
import helper_functions

G = helper_functions.load_graph("condmat")

total = 0
for i in range(10):
	score = main.k_fold_train_and_test(G.copy(), method = "lp", num_trials = 500, parameter = 0.04)
	print("Average AUC: {:.5f}".format(score))
	total += score
print(total / 10)