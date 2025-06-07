import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve


# designing my model
class PimaClassifier(nn.Module):
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


def train_model():
    dataset = np.loadtxt('dataset\diabetes_trainvalidate.csv', delimiter=',')  # 2d array rows,columns

    X = dataset[:, 0:8]  # select all columns, and the first 7 rows
    y = dataset[:, 8]  # select all columns and the last row

    # splitting my training dataset for training and validating
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=42)

    # converting numpy's 64-bit floats to torch's 32-bits floats
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_vali = torch.tensor(X_vali, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_vali = torch.tensor(y_vali, dtype=torch.float32).reshape(-1, 1)

    # loading my model and printing it
    model = PimaClassifier()

    print(f"1. This is my model with 3 layers: \n \n{model} \n \n")

    # preparing the model for training
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training it
    n_epochs = 100
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

        if epoch % 20 == 0 or epoch == (n_epochs - 1):
            print(f'Finished epoch {epoch}, latest loss {loss}')

    # validating my model (also, getting probabilities for ROC curve plotting)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_vali)

    accuracy = (y_pred.round() == y_vali).float().mean()
    print(f"\n \n 2. my model's validating Accuracy is: {accuracy} \n")

    # drawing my ROC curve
    fpr, tpr, thresholds = roc_curve(y_vali.numpy(), y_pred.numpy())  # converting back to numpy to avoid weaird issues
    roc_auc = roc_auc_score(y_vali.numpy(), y_pred.numpy())

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='diabetes')
    display.plot()
    plt.show()

    # Training done, saving my model's weights
    torch.save(model.state_dict(), "diabetes_pimamodel.pth")

def mccv():
    dataset = np.loadtxt('dataset\diabetes_trainvalidate.csv', delimiter=',')  # 2d array rows,columns

    X = dataset[:, 0:8]  # select all columns, and the first 7 rows
    y = dataset[:, 8]  # select all columns and the last row

    # loading my model and printing it
    model = PimaClassifier()

    # preparing the model for training
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_splits = 10
    metrics = {"accuracy":[],"ROCAUC":[]}

    for i in range(n_splits):
        # splitting my training dataset for training and validating
        X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=None)

        # converting numpy's 64-bit floats to torch's 32-bits floats
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_vali = torch.tensor(X_vali, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_vali = torch.tensor(y_vali, dtype=torch.float32).reshape(-1, 1)

        # training it
        n_epochs = 100
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

        # validating my model (also, getting probabilities for ROC curve plotting)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_vali)

        accuracy = (y_pred.round() == y_vali).float().mean()
        roc_auc = roc_auc_score(y_vali.numpy(), y_pred.numpy())
        metrics.get("accuracy").append(accuracy)
        metrics.get("ROCAUC").append(roc_auc)

    meanA = np.mean(metrics.get("accuracy"))
    stdA = np.std(metrics.get("accuracy"))
    meanAUC = np.mean(metrics.get("ROCAUC"))
    stdAUC = np.std(metrics.get("ROCAUC"))

    print("_____Evaluating MCCV____")

    print(f"Accuracy: [{metrics.get('accuracy')[:11]},...]")
    print(f"ROC AUC: [{metrics.get('ROCAUC')[:11]},...]")

    print(f"Mean Accuracy: {meanA:.3f} ± {stdA:.3f}")
    print(f"Mean ROC Area under the curve: {meanAUC:.3f} ± {stdAUC:.3f}")

if __name__ == "__main__":
    train_model()
    mccv()