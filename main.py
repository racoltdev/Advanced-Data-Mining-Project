import sys
from os import path

import model

if __name__ == "__main__":
	data_file = ""
	if len(sys.argv) < 2:
		# Find dataset file dependent on location of this executable, not the user's CWD
		cwd = path.dirname(path.realpath(__file__))
		default_data_file = path.join(cwd, "datasets", "enhanced_anxiety_dataset.csv")
		data_file = default_data_file
	else:
		data_file = path.abspath(sys.argv[1])

	model.mlr(data_file)
	model.decision_tree(data_file)
