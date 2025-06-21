import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve, confusion_matrix


# designing my model
class PimaClassifier(nn.Module):
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
    n_epochs = 150
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

    n_folds = 10
    metrics = {"accuracy":[],"ROCAUC":[]}

    for i in range(n_folds):
        # splitting my training dataset for training and validating
        X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=None)

        # converting numpy's 64-bit floats to torch's 32-bits floats
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_vali = torch.tensor(X_vali, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        y_vali = torch.tensor(y_vali, dtype=torch.float32).reshape(-1, 1)

        #variables tracker for future visualizations (e.g train loss vs vali loss curves...)
        train_loss = []
        vali_loss = []

        # Creating a new model at every iteration is important
        model = PimaClassifier()
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # training iteration
        n_epochs = 150
        batch_size = 10

        for epoch in n_epochs:

            loss_accumulator=0

            for i in range(0, len(X_train), batch_size):
                Xbatch = X_train[i:i + batch_size]
                y_pred = model(Xbatch)
                ybatch = y_train[i:i + batch_size]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_accumulator+=loss.item()

            #find the average train loss after each epoch
            train_loss.append(loss_accumulator / (len(X_train) / batch_size))

            #validation loss after each epoch
            model.eval()
            with torch.no_grad():
                y_vali_pred = model(X_vali)
                val_loss = loss_fn(y_vali_pred, y_vali)
                vali_loss.append(val_loss.item())


        # #plotting Idealized error curves
        # plt.plot(train_loss, label='Training Loss')
        # plt.plot(vali_loss, label='Validation Loss')
        # plt.title(f"Loss per Epoch / Idealized Error Curves for MCCV: split {epoch}")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        # validating my mccv model (also, getting probabilities for ROC curve plotting)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_vali)

    #Confusion matrix display for each iteration of my mccv for debugging
        cm= confusion_matrix(y_vali.numpy(), y_pred.round())
        TN, FP, FN, TP = cm.ravel()
        # disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        # disp.plot()

        # labels=[['TN','FP'],['FN','TP']]
        # for i1 in range(2):
        #     for j in range(2):
        #         plt.text(j, i1, f"\n\n{labels[i1][j]}", ha='center', va='center', fontsize=12, color='black')
        # plt.title(f"Confusion Matrix: split {epoch}")
        # plt.tight_layout()
        # plt.show()

        accuracy = (y_pred.round() == y_vali).float().mean().numpy()
        rfpr, rtpr, thresholds = roc_curve(y_vali.numpy(), y_pred.numpy())
        roc_auc = roc_auc_score(y_vali.numpy(), y_pred.numpy())
        metrics.get("accuracy").append(accuracy)
        metrics.get("ROCAUC").append(roc_auc)
        metrics.get("rfpr").append(rfpr)
        metrics.get("rtpr").append(rtpr)
        metrics.get("fpr").append(FP / (FP + TN))
        metrics.get("tpr").append(TP / (TP + FN))
        metrics.get("ppv").append(TP / (TP + FP))
        metrics.get("npv").append(TN / (TN + FN))


#plotting Idealized error curves to checkup on overfitting
    plt.plot(train_loss, label='Training Loss')
    plt.plot(vali_loss, label='Validation Loss')
    plt.title(f"Loss per Epoch / Idealized Error Curves for MCCV")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#finding the mean of all my mccv folds
    meanA = np.mean(metrics.get("accuracy"))
    stdA = np.std(metrics.get("accuracy"))
    meanAUC = np.mean(metrics.get("ROCAUC"))
    stdAUC = np.std(metrics.get("ROCAUC"))

    #other binary classification metrics
    meanTPR = np.mean(metrics["tpr"])
    stdTPR = np.std(metrics["tpr"])
    meanFNR = 1 - meanTPR
    stdFNR = stdTPR  # because FNR = 1 - TPR, std is the same

    meanFPR = np.mean(metrics["fpr"])
    stdFPR = np.std(metrics["fpr"])
    meanTNR = 1 - meanFPR
    stdTNR = stdFPR

    meanPPV = np.mean(metrics.get("ppv"))
    stdPPV = np.std(metrics.get("ppv"))
    meanFDR = 1 - meanPPV
    stdFDR = stdPPV

    meanNPV = np.mean(metrics.get("npv"))
    stdNPV = np.std(metrics.get("npv"))
    meanFOR = 1 - meanNPV
    stdFOR = stdNPV



#Plotting a mean ROC
        ###Standardizing my data (to review)
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []

    for fpr, tpr in zip(metrics.get("rfpr"), metrics.get("rtpr")):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_interp.append(interp_tpr)

    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr = np.std(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0  # ensure ROC ends at (1,1)

        ###plot
    plt.figure(figsize=(8,6))
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f"Mean ROC (AUC = {meanAUC:.3f} ± {stdAUC:.3f})")

    # Fill between mean_tpr ± std_tpr
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='±1 std dev')

    # Reference diagonal line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    #individual ROC curves
    for fpr, tpr in zip(metrics.get("rfpr"), metrics.get("rtpr")):
        plt.plot(fpr, tpr, 'b', alpha=0.15)


    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve with MCCV')
    plt.legend(loc='lower right')
    plt.show()

# Training done, saving my model's weights
    #torch.save(model.state_dict(), "diabetes_pimamodel.pth")



    print("_____Evaluating MCCV____")

    print(f"Accuracy: [{metrics.get('accuracy')[:11]},...]")
    print(f"ROC AUC: [{metrics.get('ROCAUC')[:11]},...]")

    print(f"Mean Accuracy: {meanA:.3f} ± {stdA:.3f}")
    print(f"Mean ROC Area under the curve: {meanAUC:.3f} ± {stdAUC:.3f}")

    #displaying the binary classification metrics
    summary = {
    "Metric": [
        "Sensitivity", "FNR",
        "Specificity", "FPR",
        "Precision", "FDR",
        "NPV", "FOR"],
    "Mean": [
        meanTPR, meanFNR,
        meanTNR, meanFPR,
        meanPPV, meanFDR,
        meanNPV, meanFOR],
    "Std Dev": [
        stdTPR, stdFNR,
        stdTNR, stdFPR,
        stdPPV, stdFDR,
        stdNPV, stdFOR]
    }

    # Create a DataFrame
    metrics_df = pd.DataFrame(summary).set_index('Metric')
    print("\n\n")
    print(metrics_df.round(3))

if __name__ == "__main__":
    #train_model()
    mccv()