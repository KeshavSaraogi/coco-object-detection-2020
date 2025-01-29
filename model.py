import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

# Define paths
s3_bucket = "images-data-coco"
train_images = f"s3://{s3_bucket}/extracted/train2017/"
val_images = f"s3://{s3_bucket}/extracted/val2017/"
train_annotations = f"s3://{s3_bucket}/extracted/annotations_trainval2017/instances_train2017.json"
val_annotations = f"s3://{s3_bucket}/extracted/annotations_trainval2017/instances_val2017.json"

# Define COCO dataset class
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, target

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

# Load Dataset
train_dataset = CocoDataset(train_images, train_annotations, transform=transform)
val_dataset = CocoDataset(val_images, val_annotations, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Load Pretrained Faster R-CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

# Define Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 5

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Save Model to S3
torch.save(model.state_dict(), "fasterrcnn_coco.pth")
print("Model training complete! Uploading to S3...")

# Upload to S3
import boto3
s3 = boto3.client('s3')
s3.upload_file("fasterrcnn_coco.pth", s3_bucket, "models/fasterrcnn_coco.pth")

print("Model uploaded to S3: s3://images-data-coco/models/fasterrcnn_coco.pth")
