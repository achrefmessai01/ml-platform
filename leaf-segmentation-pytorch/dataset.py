import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
from pycocotools.coco import COCO


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with image-mask pairs."""

    def __init__(self, root_dir, transform=None, target_transform=None, filter_empty=True):
        self.coco = COCO(os.path.join(root_dir, "_annotations.coco.json"))
        self.img_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Cache category IDs (computed once instead of per sample)
        self.cat_ids = self.coco.getCatIds()
        
        # Pre-build image ID list for faster indexing
        all_img_ids = list(self.coco.imgs.keys())
        
        # Filter out images with empty masks if requested
        if filter_empty:
            self.img_ids = []
            filtered_count = 0
            for img_id in all_img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
                if len(ann_ids) > 0:  # Only keep images with annotations
                    self.img_ids.append(img_id)
                else:
                    filtered_count += 1
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} images with empty masks from {root_dir}")
        else:
            self.img_ids = all_img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        image = decode_image(os.path.join(self.img_dir, img_info["file_name"]))
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)

        # Create mask more efficiently
        if len(anns) == 0:
            # Empty mask
            mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
        else:
            # Create mask from first annotation
            mask = self.coco.annToMask(anns[0])
            # Merge remaining annotations if any
            for ann in anns[1:]:
                mask = np.maximum(mask, self.coco.annToMask(ann))

        # Convert mask to tensor, add channel dimension, and normalize to [0, 1]
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        # Normalize if needed (handles both 0-1 and 0-255 ranges)
        if mask.max() > 1.0:
            mask = mask / 255.0

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            # Re-normalize after transform if needed
            if mask.max() > 1.0:
                mask = mask / 255.0

        return image, mask
