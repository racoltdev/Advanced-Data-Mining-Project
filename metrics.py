from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import somersd
import math

class Metrics:
	mse = 0
	rmse = 0
	r2 = 0
	somers = 0
	somers_pvalue = 0

	def __init__(self, y_test, y_predict):
		self.mse = mean_squared_error(y_test, y_predict)
		self.rmse = math.sqrt(self.mse)
		self.r2 = r2_score(y_test, y_predict)
		self.somers, self.somers_pvalue = self.calc_somers(y_test, y_predict)
		self.somers_pvalue = 1.0

	def print(self):
		print(f"Mean Squared Error (MSE): {self.mse:.4f}")
		print(f"Root Mean Squared Error (RMSE): {self.rmse:.4f}")
		print(f"R² Score: {self.r2:.4f}")
		print(f"Somers' D statistic: {self.somers:.4f}")
		print(f"Somers' D pvalue: {self.somers_pvalue:.4f}")

	def calc_somers(self, y_test, y_predict):
		y_predict = y_predict.reshape(-1, 1)
		discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
		discrete_predict = discretizer.fit_transform(y_predict)
		discrete_predict = [x[0] for x in discrete_predict]
		somers = somersd(y_test, discrete_predict)
		return somers.statistic, somers.pvalue


