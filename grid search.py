import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# trying 3 different layer models
class Classifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


class Classifier2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.LeakyReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.LeakyReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


class Classifier3(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.SiLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.SiLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


# ......................

dataset = np.loadtxt('dataset\diabetes_trainvalidate.csv', delimiter=',')  # 2d array rows,columns

X = dataset[:, 0:8]  # select all columns, and the first 7 rows
y = dataset[:, 8]  # select all columns and the last row

grid = {"lrate": [0.001, 0.01],
        "epoch": [100, 150, 200],
        "models": [Classifier1(), Classifier2(), Classifier3()]}

# preparing grid iterations
combinations = list(product(*grid.values()))  # Each tuple has exactly one value from each list, in the same order
param_names = list(grid.keys())  # preserves the order of keys

results = []

for architecture in combinations:
    parameters = dict(zip(param_names, architecture))  # create a dict of paramater names and values from a tuple

    n_splits = 5
    metrics = {"accuracy": [], "ROCAUC": [], "tpr": []}

    # run the MCCV
    for k in range(n_splits):

        # model
        model = parameters.get('models')
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=parameters.get('lrate'))

        # splitting my training dataset for training and validating
        X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=None)

        X_train = torch.tensor(X_train,
                               dtype=torch.float32)  # converting numpy's 64-bit floats to torch's 32-bits floats
        X_vali = torch.tensor(X_vali, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_vali = torch.tensor(y_vali, dtype=torch.float32).reshape(-1, 1)

        # TRAINING
        n_epochs = parameters.get('epoch')
        batch_size = 10

        for epoch in range(n_epochs):
            for i in range(0, len(X_train), batch_size):
                Xbatch = X_train[i:i + batch_size]
                y_pred = model(Xbatch)
                ybatch = y_train[i:i + batch_size]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_vali)

        accuracy = (y_pred.round() == y_vali).float().mean().numpy()
        metrics.get("accuracy").append(accuracy)

        roc_auc = roc_auc_score(y_vali.numpy(), y_pred.numpy())
        metrics.get("ROCAUC").append(roc_auc)

        cm = confusion_matrix(y_vali.numpy(), y_pred.round())
        TN, FP, FN, TP = cm.ravel()
        metrics.get("tpr").append(TP / (TP + FN))

    # saving results for models comparison
    meanA = np.mean(metrics.get("accuracy"))
    stdA = np.std(metrics.get("accuracy"))
    meanAUC = np.mean(metrics.get("ROCAUC"))
    stdAUC = np.std(metrics.get("ROCAUC"))
    meanTPR = np.mean(metrics["tpr"])
    stdTPR = np.std(metrics["tpr"])

    meanperf = {"params": architecture,
                "accuracy": f"{meanA:.3f} ± {stdA:.3f}",
                "ROCAUC": f"{meanAUC:.3f} ± {stdAUC:.3f}",
                "tpr": f"{meanTPR:.3f} ± {stdTPR:.3f}"}
    results.append(meanperf)

# structuring and exporting results
comp_table = pd.DataFrame(results)
print(comp_table)
comp_table.to_excel("results_gridsearch.xlsx", index=False)