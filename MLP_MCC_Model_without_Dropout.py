import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
#Load dataset
df = pd.read_csv("synthetic_to_train.csv")
#print(df.head())

# Separate features and target
X = df.drop(columns=["Output_Class"])
y = df["Output_Class"]
#print(f"Shape of features(X) is {X.shape}")
#print(f"Shape of label(y) is {y.shape}")


#---------------------#
#Encode Target label is labels is a string(sklearn.preprocessing import LabelEncoder())
#----------------------#

#Scale features
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

#Split the data set in to training, validation and testing
#First split: train vs temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

#Second split: val vs test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# Convert to PyTorch tensors
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Confirm shapes
print(f"Train set: {X_train_tensor.shape}")
print(f"Validation set: {X_val_tensor.shape}")
print(f"Test set: {X_test_tensor.shape}")

#Model MLP
import torch.nn as nn

class MLP_MCC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_MCC, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x):
        return self.network(x)
    

#Instantiate Model
input_dim = X_train_tensor.shape[1]
hidden_dim = 32
output_dim = 4

model_MCC = MLP_MCC(input_dim, hidden_dim, output_dim)
print(model_MCC)


#Loss Fundtion and Optimizer
import torch.optim as optim
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_MCC.parameters(), lr =0.01)

#Store loses
train_losses_MCC = []
val_losses_MCC = []

#Training Loop
epochs = 1000
for epoch in range(epochs):
    #Training phase
    model_MCC.train()
    optimizer.zero_grad()
    outputs = model_MCC(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses_MCC.append(loss.item())

#Validation phase
    model_MCC.eval()
    with torch.no_grad():
        val_outputs = model_MCC(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses_MCC.append(val_loss.item())
    

#Option: print every 50 epochs
#if (epoch+1)%50 ==0:
#    print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}, val_loss = {val_loss.item():.4f}")

#Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(train_losses_MCC, label="Training Loss")
plt.plot(val_losses_MCC, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (No Regularization)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Loss (No Regularization).png")
plt.show()

