import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

g = 9.81
L = 1.0
t_min, t_max = 0, 10
n_train = 100

t_train = torch.linspace(t_min, t_max, n_train, requires_grad=True).view(-1, 1)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Salida: ángulo θ
        )

    def forward(self, t):
        return self.net(t)

def physics_loss(model, t):
    theta = model(t)  # Predicción de θ
    theta_t = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t, t, grad_outputs=torch.ones_like(theta_t), create_graph=True)[0]
    loss = torch.mean((theta_tt + (g / L) * torch.sin(theta))**2)
    return loss

model = PINN()

optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 5000
loss_history = []
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = physics_loss(model, t_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Physics Loss")
plt.show()

t_test = torch.linspace(t_min, t_max, 100).view(-1, 1)
theta_pred = model(t_test).detach().numpy()

plt.plot(t_test.numpy(), theta_pred, label="Predicción PINN")
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo θ (rad)")
plt.legend()
plt.show()
