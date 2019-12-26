
import numpy as np
from sklearn import metrics
y = np.array([0, 1])
pred = np.array([1, 1])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
print(metrics.auc(fpr, tpr))
