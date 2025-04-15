import sys
import os

if __name__ == "__main__":
	data_file = ""
	if len(sys.argv) < 2:
		cwd = os.path.dirname(os.path.realpath(__file__))
		default_data_file = os.path.join(cwd, "datasets", "enhanced_anxiety_dataset.csv")
		data_file = default_data_file
	else:
		data_file = os.path.abspath(sys.argv[1])

	print(data_file)
