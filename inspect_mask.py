import torchvision.datasets as datasets
import numpy as np
from PIL import Image

print("--- Inspecting Original Mask ---")

# Load the dataset *without* your custom class
full_dataset = datasets.OxfordIIITPet(
    root=".", 
    split="trainval", 
    target_types="segmentation", 
    download=True
)

# Get the first sample
image, mask = full_dataset[0]

print(f"Original mask type: {type(mask)}")
print(f"Original mask mode (if PIL): {mask.mode}")
print(f"Original mask size: {mask.size}")

# Convert to numpy to see the values
mask_np = np.array(mask)

print(f"Mask numpy dtype: {mask_np.dtype}")
print(f"Unique pixel values in mask: {np.unique(mask_np)}") # <-- This is the key!

# Save it to look at
mask.save("original_mask_sample.png")
print("Saved 'original_mask_sample.png' for inspection.")
print("---------------------------------")


import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as T
from torchvision.transforms.v2.functional import resize, to_image, to_dtype
import numpy as np
import torchvision.utils as vutils
from PIL import Image

print("--- Mask Inspection Script ---")

# --- 1. Load the dataset ---
print("Loading original dataset...")
full_dataset = datasets.OxfordIIITPet(
    root=".", 
    split="trainval", 
    target_types="segmentation", 
    download=False # Assumes it's already downloaded
)

# --- 2. Get one sample ---
# We'll find a Beagle, as they are in your filter
beagle_idx = full_dataset.class_to_idx['Beagle']
sample_idx = -1
for i in range(len(full_dataset)):
    if full_dataset._labels[i] == beagle_idx:
        sample_idx = i
        break
        
if sample_idx == -1:
    print("Error: Could not find a sample image.")
    exit()

print(f"Loading sample {sample_idx}...")
image, mask = full_dataset[sample_idx] # mask is a PIL Image

# --- 3. Apply your __getitem__ logic EXACTLY ---
print("Applying __getitem__ logic...")
target_size = (512, 512)

# --- Resize ---
mask = resize(mask, target_size, interpolation=T.InterpolationMode.NEAREST)

# --- Process mask (The logic in question) ---
mask_np = np.array(mask, dtype=np.uint8)  # Load as integer

# Create binary mask: 1.0 where mask is (1 OR 2), 0.0 otherwise (where mask is 3)
print("Applying np.where((mask_np == 1) | (mask_np == 2))...")
processed_mask_np = np.where(
    (mask_np == 2),  # This *should* fill the mask
    0.0,  # Background
    1.0   # Forground
).astype(np.float32)

mask_tensor = torch.from_numpy(processed_mask_np).unsqueeze(0)  # (1, 512, 512)
print(f"Final mask tensor shape: {mask_tensor.shape}")
print(f"Final mask tensor unique values: {torch.unique(mask_tensor)}")


# --- 4. Save the resulting mask tensor as an image ---
vutils.save_image(mask_tensor, "inspected_mask_output.png")

print("\n--- DONE ---")
print("Saved 'inspected_mask_output.png'.")
print("Please open this file. Is it a FILLED shape or just an OUTLINE?")

# --- Bonus: Let's check the 'outline-only' logic ---
print("\n--- Bonus Check ---")
print("Applying 'outline-only' logic (mask_np == 2)...")
outline_only_np = np.where(
    (mask_np == 2),  # This is the buggy logic
    1.0,
    0.0
).astype(np.float32)
outline_tensor = torch.from_numpy(outline_only_np).unsqueeze(0)
vutils.save_image(outline_tensor, "inspected_mask_OUTLINE_ONLY.png")
print("Saved 'inspected_mask_OUTLINE_ONLY.png' for comparison.")