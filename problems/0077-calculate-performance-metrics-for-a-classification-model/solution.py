import numpy as np

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	actual = np.array(actual, dtype=np.int_)
	predicted = np.array(predicted, dtype=np.int_)
	tp = np.sum((actual==1)&(predicted==1))
	fp = np.sum((actual==0)&(predicted==1))
	fn = np.sum((actual==1)&(predicted==0))
	tn = np.sum((actual==0)&(predicted==0))
	confusion_matrix = [[tp, fn], [fp, tn]]
	accuracy = (tp+tn) / (tp+tn+fp+fn)
	precision = tp / (tp+fp)
	recall = tp / (tp+fn)
	specificity = tn / (tn+fp)
	negativePredictive = tn / (tn+fn)
	f1 = 2 *(precision*recall) / (precision+recall)
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
