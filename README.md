# Developing a Neural Network Classification Model
## NAME: SANJITH.R
## REG NO: 212223230191
## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![Screenshot 2025-03-24 103025](https://github.com/user-attachments/assets/8f48c5c3-c8a5-468f-9352-da3fe6bbd7e7)



## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.
### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.
### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).


### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.


## PROGRAM

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Dataset ---
# Make sure 'customers.csv' is in your Colab files
data = pd.read_csv("/content/customers.csv")

# --- 2. Data Preprocessing ---
# Drop ID column as it's not useful for classification
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

# Encode categorical variables
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable (Segmentation: A, B, C, D -> 0, 1, 2, 3)
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])

# Split features and target
X = data.drop(columns=["Segmentation"]).values
y = data["Segmentation"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# --- 3. Neural Network Architecture ---
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)  # 4 output classes (A, B, C, D)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation needed here, CrossEntropyLoss applies Softmax internally
        return x

# Initialize Model
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 4. Training Loop ---
loss_history = []

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Train for 100 Epochs
train_model(model, train_loader, criterion, optimizer, epochs=100)

# --- 5. Evaluation ---
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions) * 100
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])

print("\n" + "="*30)
print("Name: SANJITH R")
print("Reg No: 212223230191")
print("="*30)

print(f'\nTest Accuracy: {accuracy:.2f}%')
print("Classification Report:\n", class_report)

# --- 6. Plots ---

# Plot 1: Training Loss (Added for you)
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss')
plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot 2: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# --- 7. Prediction for a sample input ---
sample_input = X_test_tensor[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    actual_class_label = label_encoder.inverse_transform([y_test_tensor[12].item()])[0]

print(f'\nPredicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {actual_class_label}')


```



## Dataset Information

![Screenshot 2025-03-17 113110](https://github.com/user-attachments/assets/876b68a3-3a95-4f72-84db-a2f05312f01f)



## OUTPUT

## EPOCH

<img width="507" height="357" alt="image" src="https://github.com/user-attachments/assets/0a816190-7180-4073-b0fd-15626d3acf01" />
<img width="844" height="561" alt="image" src="https://github.com/user-attachments/assets/a1a1ca8b-c559-47ea-baf1-6d877b662aca" />


### Confusion Matrix

<img width="650" height="546" alt="image" src="https://github.com/user-attachments/assets/038d6539-ad88-4b16-94a6-82025a9f77b4" />



### Classification Report


<img width="852" height="279" alt="image" src="https://github.com/user-attachments/assets/f43da371-299b-4bd4-a5c0-ca1cedbb5c03" />



## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
