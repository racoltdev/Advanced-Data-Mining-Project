import matplotlib.pyplot as plt
import seaborn
import pandas

def plot_correlation(df):
	pearson_correlation = df.corr(method="pearson", numeric_only=True)
	kendall_correlation = df.corr(method="kendall", numeric_only=True)

	plt.title("Pearson Correlation")
	seaborn.heatmap(pearson_correlation, cmap="flare", annot=True)
	plt.xticks(rotation=35, ha='right')
	plt.show()

	plt.title("Kendall Correlation")
	seaborn.heatmap(kendall_correlation, cmap="flare", annot=True)
	plt.xticks(rotation=35, ha='right')
	plt.show()

df = pandas.read_csv("datasets/enhanced_anxiety_dataset.csv")
plot_correlation(df)

