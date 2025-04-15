from sklearn.metrics import mean_squared_error, r2_score
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
		self.somers = 0
		self.somers_pvalue = 1

	def print(self):
		print(f"Mean Squared Error (MSE): {self.mse:.4f}")
		print(f"Root Mean Squared Error (RMSE): {self.rmse:.4f}")
		print(f"R² Score: {self.r2:.4f}")

