import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms.v2 as T
from torchvision.transforms.v2.functional import to_tensor, resize, normalize
from PIL import Image
import numpy as np
import os
import random 

from torch.optim.lr_scheduler import ReduceLROnPlateau
#image saving util
import torchvision.utils as vutils



if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" #apple 
else:
    DEVICE = "cpu"

LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_EPOCHS = 3
IMG_HEIGHT = 512
IMG_WIDTH = 512 




#The UNET MODEL

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels): 
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )


    def forward(self, x): 
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        #Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc5 = DoubleConv(512, 1024) #bottle neck

        #Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x): 
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)
        b = self.enc5(p4)
        u4 = self.up4(b)
        c4 = torch.cat([u4, s4], dim=1)
        d4 = self.dec4(c4)
        u3 = self.up3(d4)
        c3 = torch.cat([u3, s3], dim=1)
        d3 = self.dec3(c3)
        u2 = self.up2(d3)
        c2 = torch.cat([u2, s2],dim=1)
        d2 = self.dec2(c2)
        u1 = self.up1(d2)
        c1 = torch.cat([u1, s1], dim=1)
        d1 = self.dec1(c1)
        output = self.out_conv(d1)
        return output #raw logits






#DATASET, loading, transformations etc. 
class OxfordPetDataset(Dataset):
    """
    Custom Dataset for the Oxford-IIIT Pet Dataset (Segmentation).
    This class handles loading, transforming, and remapping the masks.
    """
    def __init__(self, root=".", split="trainval", target_size=(512, 512)):
        assert target_size[0] == target_size[1], "Target size must be square"
        self.target_size = target_size
        
        full_dataset = datasets.OxfordIIITPet(
            root=root, 
            split=split, 
            target_types="segmentation", 
            download=True,
            transform=None,
            target_transform=None
        )
        
        #Filter dataset to specified classes
        beagle_class_index = full_dataset.class_to_idx['Beagle']
        abyssinian_index = full_dataset.class_to_idx["Abyssinian"]
        
        filter_ids = [beagle_class_index, abyssinian_index]
        self.filtered_indices = []
        
        # Iterate over the labels and find matching indices
        for i in range(len(full_dataset)):
            if full_dataset._labels[i] in filter_ids:
                self.filtered_indices.append(i)
        
        #Shuffle the filtered list
        if split == "trainval":
            random.shuffle(self.filtered_indices)
                
        print(f"Found {len(self.filtered_indices)} images in '{split}' split (after filtering).")
        
        # Store the full dataset to access it by filtered index
        self.full_dataset = full_dataset
        
        
        #augmentation pipeline
        self.do_augmentation = (split == "trainval")
        if self.do_augmentation:
            print("Augmentation is ENABLED for this split.")
            # 1. Geometric transforms (applied to Image AND Mask)
            self.geometric_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
            ])
            # 2. Image-only transforms (applied to Image ONLY)
            self.image_only_transforms = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.RandomGrayscale(p=0.1)
            ])
        else:
            print("Augmentation is DISABLED for this split.")
            
    
    def __len__(self):
        return len(self.filtered_indices)
        
    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        
        # Load the PIL image and PIL mask from the full dataset
        image, mask = self.full_dataset[original_idx]
        
        #Apply Augmentations
        if self.do_augmentation:
            # Apply geometric transforms to both
            image, mask = self.geometric_transforms(image, mask)
            # Apply color transforms to image only
            image = self.image_only_transforms(image)

        #Resize to the correct dims for UNET (must be devisable by 16)
        image = resize(image, self.target_size, interpolation=T.InterpolationMode.BILINEAR)
        mask = resize(mask, self.target_size, interpolation=T.InterpolationMode.NEAREST)
        
        #Convert to Tensor
        image_tensor = to_tensor(image) # Scales image to [0.0, 1.0]
        
        #Process Mask
        mask_np = np.array(mask) 
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_tensor[mask_tensor == 2] = 0
        mask_tensor[mask_tensor == 3] = 0
        mask_tensor = mask_tensor.unsqueeze(0).float()
        
        #Normalize Image
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        image_tensor = normalize(image_tensor, mean=self.mean, std=self.std)
        
        return image_tensor, mask_tensor




# --- FIXED: Add Dice Loss and DiceBCELoss ---
class DiceLoss(nn.Module):
    """Calculates the Dice Loss (IoU) for segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred is raw logits, apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Flatten both
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # We want to MINIMIZE the loss, so we return 1 - Dice
        return 1 - dice

class DiceBCELoss(nn.Module):
    """Combines Dice Loss and BCE Loss."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        
        # Combine the two losses
        total_loss = (self.dice_weight * dice_loss) + (self.bce_weight * bce_loss)
        return total_loss


def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    
    for _, (data, targets) in enumerate(loader):
        data = data.to(device)
        targets = targets.to(device)
        
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    print(f"  Epoch Training Loss: {avg_loss:.4f}")


def save_predictions_as_imgs(
    loader, model, epoch, folder="saved_images/", device="cuda"
):
    print(f"Saving prediction images to {folder}...")
    os.makedirs(folder, exist_ok=True)
    model.eval()
    
    data, targets = next(iter(loader))
    data = data.to(device=device)
    targets = targets.to(device=device)
    
    with torch.no_grad():
        predictions = model(data)
        
    predictions_proba = torch.sigmoid(predictions)
    predictions_binary = (predictions_proba > 0.5).float()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    data_unnorm = data * std + mean
    data_unnorm = torch.clamp(data_unnorm, 0, 1)
    
    targets_rgb = targets.repeat(1, 3, 1, 1)
    predictions_rgb = predictions_binary.repeat(1, 3, 1, 1)
    
    comparison_grid = torch.cat([data_unnorm, targets_rgb, predictions_rgb], dim=0)
    
    vutils.save_image(
        comparison_grid,
        f"{folder}/comparison_epoch_{epoch+1}.png",
        nrow=BATCH_SIZE
    )
    
    model.train() 
    

def main():
    print(f"Using device: {DEVICE}")
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # --- FIXED: Use the robust DiceBCELoss ---
    loss_fn = DiceBCELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=2,
    )
    
    # --- 4. Create Datasets and DataLoaders ---
    train_dataset = OxfordPetDataset(
        root=".",
        split="trainval",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    val_dataset = OxfordPetDataset(
        root=".",
        split="test",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )

    # --- 5. The Training Loop ---
    print(f"Starting training, lr = {LEARNING_RATE}, Batchsize= {BATCH_SIZE}, Epochs={NUM_EPOCHS}")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        
        # --- Validation Step ---
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for data, targets in val_loader:
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item()
                
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"  Epoch Validation Loss: {avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)

        save_predictions_as_imgs(
            val_loader, model, epoch, folder="saved_images/", device=DEVICE
        )

    print("Training complete!")

    return model 

if __name__ == "__main__":
    #train model 
    model = main()


    #test model . 
    train_dataset = OxfordPetDataset(
        root=".",
        split="trainval",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False
    )

    #run prediction on 1 batch of training data to vis some results. 
    save_predictions_as_imgs(train_loader, model, epoch=NUM_EPOCHS, folder="predictions/", device=DEVICE)

        
