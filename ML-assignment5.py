import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps


# --- 1. Model Definition ---
class DigitMLP(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super(DigitMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 28x28 image to 784 1D vector
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. Custom Image Prediction Function ---
def predict_custom_image(image_path, model):
    try:
        img = Image.open(image_path).convert("L")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found in the project directory.")
        return

    # Invert colors if image background is light
    if np.array(img).mean() > 128:
        img = ImageOps.invert(img)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_digit].item() * 100.0

    print("\n" + "=" * 50)
    print(f" CUSTOM IMAGE RECOGNITION RESULTS ('{image_path}')")
    print("=" * 50)
    print(f"  -> Predicted Digit : {predicted_digit}")
    print(f"  -> Model Confidence: {confidence:.2f}%")
    print("=" * 50)


# --- 3. Main Execution ---
if __name__ == "__main__":
    # Load and Train
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = DigitMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training PyTorch Digit Classification Model...")
    model.train()
    for epoch in range(1, 4):  # Quick 3 epochs for fast training
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Complete.")

    # Predict custom image in project directory
    predict_custom_image("digit.jpg", model)