import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.generator import UNet

class CustomDataset(Dataset):
    def __init__(self, humans_images_path, clothes_path, transform=None):
        self.human_images_path = humans_images_path
        self.clothes_path = clothes_path
        self.transform = transform
        self.human_image_files = os.listdir(humans_images_path)
        
    def __len__(self):
        return len(self.human_image_files)
    
    def __getitem__(self, index):
        img_file = self.human_image_files[index]
        img = cv2.imread(os.path.join(self.human_images_path, img_file))
        cloth = cv2.imread(os.path.join(self.clothes_path, img_file))
        
        if self.transform:
            img = self.transform(img)
            cloth = self.transform(cloth)
            
        return img, cloth
    
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(), # Convert to tensor
])

# Create datasets and dataloaders
train_dataset = CustomDataset('../data/train/preprocessed_humans/', '../data/train/preprocessed_clothes/', transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model, optimizer, and loss function
model = UNet(in_channels=3, out_channels=3) # Change channels as needed
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = torch.nn.L1Loss()

# Training loop
num_epochs = 10 # Set number of epochs

for epoch in range(num_epochs):
    model.train()
    for human_img, cloth_img in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(human_img) # forward pass
        loss = criterion(output, cloth_img) # calculate loss
        loss.backward() # backpropagation
        optimizer.step() # update weights
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item(): .4f}')
    
print("Training completed.")

torch.save(model.state_dict(), 'unet_model.pth')
print("Model saved as 'unet_model.pth")