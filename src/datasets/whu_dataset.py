from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import numpy as np



IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Folder not found: {directory}")
    
    files = [p for p in directory.iterdir() if p.is_file() and _is_image_file(p)]
    files.sort()

    return files

def _stem(path: Path) -> str:
    return path.stem

def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def _load_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")

def _mask_to_binary(mask: Image.Image) -> torch.Tensor:
    mask_np = np.array(mask, dtype=np.uint8 )
    mask_bin = (mask_np > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)
    return mask_tensor

class JointTransfrom:
    def __init__(self,
                 train: bool = True,
                 resize: Optional[Tuple[int, int]] = None,
                 random_hflip: float = 0.5,
                 random_vflip: float = 0.5,
                    random_rotation: bool = True,
                    normalize: bool = True,
                 
                 ):
        self.train = train
        self.resize = resize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.random_rotate = random_rotation
        self.normalize = normalize
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.to_tensor = T.ToTensor()
    
    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.resize is not None:
            img = TF.resize(img, self.resize, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.resize, interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if self.random_hflip > 0 and torch.rand(1) < self.random_hflip:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if self.random_vflip > 0 and torch.rand(1) < self.random_vflip:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            if self.random_rotate:
                k = int(torch.randint(low=0, high=4, size=(1,)).item())
                if k:
                    angle = k * 90
                    img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.NEAREST)
                    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        img_t = self.to_tensor(img)

        mask_t = self.to_tensor(mask)

        if self.normalize:
            img_t = TF.normalize(img_t, mean=self.mean, std=self.std)

        return img_t, mask_t
    
def build_transforms(
        train: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
) -> JointTransfrom:
    return JointTransfrom(
        train=train,
        resize=image_size,
        random_hflip=0.5 if train else 0.0,
        random_vflip=0.5 if train else 0.0,
        random_rotation=train,
        normalize=True,
    )

class WHUDataset(Dataset):
    def __init__(self,
                 root_dir: Union[str, Path],
                 split: str = "train",
                 transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
                 return_meta: bool = False,
                 strict_pairing: bool = True,
                 ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {self.split}. Supported splits are 'train', 'val', 'test'.")
        
        self.images_dir = self.root_dir / self.split / "image"
        self.masks_dir = self.root_dir / self.split / "label"

        self.transform = transform
        self.return_meta = return_meta
        self.strict_pairing = strict_pairing

        self.samples = self._build_index()
        super().__init__()

    def _build_index(self) -> List[Tuple[Path, Path]]:
        img_files = _list_images(self.images_dir)
        if not img_files:
            raise RuntimeError(f"No images found in {self.images_dir}")
        
        label_files = _list_images(self.masks_dir)
        if not label_files:
            raise RuntimeError(f"No masks found in {self.masks_dir}")
        
        label_map: Dict[str, Path] = {_stem(p): p for p in label_files}

        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []

        for img_path in img_files:
            stem = _stem(img_path)
            mask_path = label_map.get(stem)
            if mask_path is None:
                missing.append(stem)
                continue
            pairs.append((img_path, mask_path))

        if self.strict_pairing and missing:
            raise RuntimeError(f"Missing masks for images: {missing}")
        
        if not pairs:
            raise RuntimeError("No valid image-mask pairs found.")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index:int):
        img_path, mask_path = self.samples[index]

        img = _load_rgb(img_path)
        mask = _load_mask(mask_path)

        if img.size != mask.size:
            raise RuntimeError(f"Image and mask size mismatch for {img_path} and {mask_path}")
        
        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask)
        else:
            img_t = T.ToTensor()(img)
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask_t = _mask_to_binary(mask)
        
        if self.return_meta:
            meta = {
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "original_size": img.size,
                "stem": img_path.stem,
                "split": self.split,
            }
            return img_t, mask_t, meta
        
        return img_t, mask_t