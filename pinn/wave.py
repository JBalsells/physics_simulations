import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros del problema
c = 1.0  # Velocidad de propagación de la onda

# Red neuronal para aproximar u(x, t)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        return self.net(inputs)

# Generación de datos de entrenamiento
N_f = 5000  # Puntos internos
N_bc = 100  # Puntos de la frontera
N_ic = 100  # Puntos de la condición inicial

t_x = torch.rand(N_f, 1) * 1.0  # t en [0,1]

x_f = torch.rand(N_f, 1) * 2.0 - 1.0  # x en [-1,1]
t_f = torch.rand(N_f, 1) * 1.0  # t en [0,1]

x_bc = torch.cat((torch.ones(N_bc, 1), -torch.ones(N_bc, 1)))
t_bc = torch.rand(N_bc * 2, 1) * 1.0

x_ic = torch.rand(N_ic, 1) * 2.0 - 1.0
t_ic = torch.zeros(N_ic, 1)
u_ic = torch.sin(torch.pi * x_ic)  # Condición inicial

def wave_loss(model):
    x_f.requires_grad = True
    t_f.requires_grad = True
    u = model(x_f, t_f)
    
    u_t = torch.autograd.grad(u, t_f, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_f, torch.ones_like(u_t), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
    
    physics_loss = torch.mean((u_tt - c**2 * u_xx) ** 2)
    
    # Condiciones de frontera
    u_bc = model(x_bc, t_bc)
    bc_loss = torch.mean(u_bc ** 2)
    
    # Condición inicial
    u_ic_pred = model(x_ic, t_ic)
    ic_loss = torch.mean((u_ic_pred - u_ic) ** 2)
    
    return physics_loss + bc_loss + ic_loss

model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = wave_loss(model)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Entrenamiento finalizado.")

# Visualización de resultados
x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.full_like(x_test, 0.5)  # Fijamos un tiempo t = 0.5

with torch.no_grad():
    u_pred = model(x_test, t_test).cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(x_test.cpu().numpy(), u_pred, label='Predicción de PINN')
plt.xlabel('x')
plt.ylabel('u(x, t=0.5)')
plt.title('Solución de la ecuación de onda')
plt.legend()
plt.grid()
plt.show()