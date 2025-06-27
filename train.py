# 🔧 Import required libraries
import torch                      # Main PyTorch library
import torch.nn as nn             # For building neural network layers
import torch.optim as optim       # For training the model (optimizers)
from model import BlackjackNet    # Import the model you defined earlier
import pickle                     # For loading the saved dataset
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets

# 📂 STEP 1: Load the dataset from the pickle file
with open("balanced_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)  # dataset = [ ([state], label), ... ]

# 🧠 STEP 2: Convert the dataset into X (inputs) and y (correct labels)
# Makes use of fancy for loop for only the 1st/2nd value is cared for
X = torch.tensor([x for x, _ in dataset], dtype=torch.float32)  # 13 input features per sample
y = torch.tensor([y for _, y in dataset], dtype=torch.long)     # Labels: 0 = hit, 1 = stand

# ✂️ STEP 3: Split the dataset into 90% training, 10% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 🏗️ STEP 4: Load the model architecture
model = BlackjackNet()  # Uses the class you defined in model.py

# ⚖️ STEP 5: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()        # Tells us how wrong the model is, is the error calculator
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusts the model to reduce the loss
#lr = learning rate
#model.paramteres() = all the internal knobs and tools

# 🔁 STEP 6: Training loop (repeat over multiple epochs)
epochs = 10  # How many times we loop over the whole dataset
for epoch in range(epochs):
    model.train()               # Set model to training mode
    optimizer.zero_grad()       # Reset gradients from the last step

    outputs = model(X_train)    # Predict actions for all training samples, X INPUT, automatically calls foward()
    loss = criterion(outputs, y_train)  # Calculate how wrong those predictions are

    loss.backward()             # Backpropagation: compute gradients
    optimizer.step()            # Update model weights using optimizer

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")  # Show progress
    #.item() turns the tensor loss into a python nmumber and of 4 decimal places

# 🧪 STEP 7: Evaluate the model on unseen test data
model.eval()  # Set model to evaluation (inference) mode, including things like turning off random neruons ? 
with torch.no_grad():  # Disable gradient tracking for performance
    predictions = model(X_test)                                # Predict on test data
    predicted_classes = torch.argmax(predictions, dim=1)       # Choose action with highest score
    accuracy = (predicted_classes == y_test).float().mean()    # Compare to actual labels
    print(f"Test Accuracy: {accuracy:.4f}")                    # Show how well model performs

# 💾 STEP 8: Save the trained model so we can use it later
torch.save(model.state_dict(), "basic_strategy_ai.pt")
print("✅ Model saved as basic_strategy_ai.pt")