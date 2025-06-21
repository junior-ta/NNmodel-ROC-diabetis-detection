import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from MC_cross_validation import PimaClassifier

# 2d array rows,columns
dataset = np.loadtxt('dataset\diabetes_testing.csv', delimiter=',')

X = dataset[:, 0:8]  # select all columns, and the first 7 rows
y = dataset[:, 8]  # select all columns and the last row

# converting numpy's 64-bit floats to torch's 32-bits floats
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# load my model and weights from the training module
model = PimaClassifier()
model.load_state_dict(torch.load("diabetes_pimamodel.pth"))  # weights (results from training)
model.eval()

print("==== TESTING RESULTS ====")

# Testing
with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == y).float().mean()
print(f"\n \n 1. achieved accuracy on this test is: {accuracy} \n")
roc_auc = roc_auc_score(y.numpy(), y_pred.numpy())
print(f"\n \n 2. achieved performance on this test is: {roc_auc} \n")

# make class predictions with the model, using 0.5 as threshold
predictions = (model(X) > 0.5).int()
for i in range(10):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))