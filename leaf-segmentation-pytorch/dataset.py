import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
import hashlib
import pickle


def _process_mask_worker(args):
    """Worker function for parallel mask processing (must be at module level for pickling)"""
    img_id, coco, cat_ids, target_transform = args
    
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    
    if len(anns) == 0:
        img_info = coco.imgs[img_id]
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    else:
        mask = coco.annToMask(anns[0])
        for ann in anns[1:]:
            mask = np.maximum(mask, coco.annToMask(ann))
    
    # Convert to tensor and normalize
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
    if mask_tensor.max() > 1.0:
        mask_tensor = mask_tensor / 255.0
    
    # Apply target transform if provided (e.g., resize to 256x256)
    if target_transform:
        mask_tensor = target_transform(mask_tensor)
    
    return img_id, mask_tensor


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with image-mask pairs."""

    def __init__(self, root_dir, transform=None, target_transform=None, filter_empty=True):
        print(f"Loading dataset from {root_dir}...")
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
        
        # Create cache directory
        cache_dir = os.path.join(root_dir, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename based on transform configuration
        transform_str = str(target_transform) if target_transform else "None"
        cache_hash = hashlib.md5(transform_str.encode()).hexdigest()[:8]
        cache_file = os.path.join(cache_dir, f'masks_{cache_hash}.pkl')
        
        # Try to load from disk cache
        if os.path.exists(cache_file):
            print(f"Loading pre-computed masks from cache...")
            with open(cache_file, 'rb') as f:
                self.cached_masks = pickle.load(f)
            # Verify cache has all needed masks
            if set(self.img_ids).issubset(set(self.cached_masks.keys())):
                total_size_mb = sum(m.element_size() * m.nelement() for m in self.cached_masks.values()) / (1024**2)
                print(f"Dataset ready with {len(self.img_ids)} samples! Cache size: {total_size_mb:.1f} MB")
                return
            else:
                print("Cache incomplete, regenerating...")
        
        # **CRITICAL OPTIMIZATION**: Pre-compute, transform, and cache all masks
        print(f"Generating {len(self.img_ids)} masks (this will be cached for future use)...")
        self.cached_masks = {}
        
        # Use multiprocessing for CPU-bound COCO annToMask operations
        from multiprocessing import Pool, cpu_count
        
        # Prepare arguments for worker processes
        worker_args = [(img_id, self.coco, self.cat_ids, target_transform) for img_id in self.img_ids]
        
        # Process masks in parallel using multiple processes
        num_workers = min(cpu_count(), 8)
        print(f"  Using {num_workers} processes for mask generation...")
        
        with Pool(processes=num_workers) as pool:
            for idx, (img_id, mask_tensor) in enumerate(pool.imap(_process_mask_worker, worker_args)):
                self.cached_masks[img_id] = mask_tensor
                
                if (idx + 1) % 50 == 0 or idx + 1 == len(self.img_ids):
                    print(f"  Processed {idx + 1}/{len(self.img_ids)} masks...")
        
        # Save to disk cache
        print(f"Saving masks to cache for future use...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cached_masks, f)
        
        # Calculate memory usage
        total_size_mb = sum(m.element_size() * m.nelement() for m in self.cached_masks.values()) / (1024**2)
        print(f"Dataset ready with {len(self.img_ids)} samples! Cache size: {total_size_mb:.1f} MB")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image and convert to float [0, 1] immediately
        image = decode_image(os.path.join(self.img_dir, img_info["file_name"]))
        image = image.float() / 255.0
        
        # Apply image transform (e.g., resize)
        if self.transform:
            image = self.transform(image)
        
        # Get pre-cached and pre-transformed mask (no clone needed - tensors are immutable in practice)
        mask = self.cached_masks[img_id]

        return image, mask
