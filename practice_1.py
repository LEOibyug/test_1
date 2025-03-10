import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  

torch.manual_seed(42)
np.random.seed(42)

input_size = 10
output_size = 2
num_samples = 1000

X = torch.randn(num_samples, input_size)
y = torch.randint(0, output_size, (num_samples,))

class MLPNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPNetwork, self).__init__()

        self.l1 = nn.Linear(input_size, 4)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(4, 8)
        self.l4 = nn.ReLU()
        self.l5 = nn.Linear(8, 16)
        self.l6 = nn.ReLU()
        self.l7 = nn.Linear(16, 4)
        self.l8 = nn.ReLU()
        self.ll = nn.Linear(4, output_size)
        
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.ll(x)
        return x

model = MLPNetwork(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

train_losses = []

for epoch in range(num_epochs):
    indices = torch.randperm(num_samples)
    
    total_loss = 0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_samples)]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, 'b-')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    test_X = torch.randn(10, input_size)
    predictions = model(test_X)
    _, predicted_classes = torch.max(predictions, 1)
    print("\n测试样本预测结果:", predicted_classes)