import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	days = np.arange(10)+1
	daily_fluc = 10 * np.sin(2*PI*days/10)
	detrended_data = consumption_data - daily_fluc
	# 2) Perform linear regression on the detrended data.
	# assyme detrended_data = m * days + b

	# m = (detrended_data[1] - detrended_data[0]) / 1
	# b = detrended_data[1] - m * days[1]  # this way it only uses 2 data points; need to use all data points!

	$
	# vector = np.polyfit(days, detrended_data, deg=1)
	# m, b = vector[0], vector[1]
	
	# another way: least squares of linear regression
	x_bar = np.mean(days)
	y_bar = np.mean(detrended_data)
	m = np.sum((days - x_bar)*(detrended_data-y_bar)) / np.sum((days-x_bar)**2)
	b = y_bar - m * x_bar

	# 3) Predict day 15's base consumption.
	day_15_base = m * 15 + b

	# 4) Add the day 15 fluctuation back.
	day_15_fluc = 10 * np.sin(2*PI*15/10)
	day_15_data = day_15_base + day_15_fluc
	# 5) Round, then add a 5% safety margin (rounded up).
	day_15_data = math.ceil(1.05*day_15_data)
	# 6) Return the final integer.
	return day_15_data
	