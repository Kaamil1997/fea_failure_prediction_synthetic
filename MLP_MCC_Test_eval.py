#Final Evaluation on Test set
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        logits = model_MCC(X_test_tensor)
        preds = torch.argmax(logits, dim=1)

    y_true = y_test_tensor.cpu().numpy()
    y_pred = preds.cpu().numpy()


    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,  average="macro")

    

    return acc, f1