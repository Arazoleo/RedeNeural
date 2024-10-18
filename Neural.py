import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Carregar e Preprocessar a Imagem
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Converter para escala de cinza
    image_np = np.array(image) / 255.0  # Normalizar para [0, 1]
    return image_np

# 2. Definir a Rede Neural (Autoencoder)
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),  # Ajuste de acordo com a dimensão da imagem
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),  # Ajuste para retornar ao tamanho original
            nn.Sigmoid(),  # Para garantir que a saída esteja entre 0 e 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. Treinar o Modelo
def train_autoencoder(model, data, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()  # Para dados normalizados
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Passar os dados pela rede
        output = model(data)
        loss = criterion(output, data)  # Comparar com a entrada
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. Avaliar e Visualizar os Resultados
def visualize_reconstruction(original, reconstructed):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagem Original')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Imagem Reconstruída')
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')

    plt.show()

# Executando o código
image_path = 'brain.jpg'  # Caminho da sua imagem comprimida
original_image_np = load_and_preprocess_image(image_path)
original_height, original_width = original_image_np.shape
data_tensor = torch.FloatTensor(original_image_np).view(-1, original_height * original_width)

# Instanciar e treinar o autoencoder
autoencoder = Autoencoder(input_size=original_height * original_width)
train_autoencoder(autoencoder, data_tensor)

# Avaliar a reconstrução
autoencoder.eval()
with torch.no_grad():
    reconstructed = autoencoder(data_tensor).numpy().reshape(original_height, original_width)

# Visualizar os resultados
visualize_reconstruction(original_image_np, reconstructed)
