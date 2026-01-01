from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

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


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    # (C,H,W) uint8, RGB
    img = read_image(str(path), mode=ImageReadMode.RGB)
    # Ensure 3 channels
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3]
    return img


def _load_mask_tensor(path: Path) -> torch.Tensor:
    # read as grayscale (1,H,W) uint8
    m = read_image(str(path), mode=ImageReadMode.GRAY)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] > 1:
        m = m[:1]
    return m


def _mask_to_binary_tensor(mask_u8: torch.Tensor) -> torch.Tensor:
    # mask_u8: (1,H,W) uint8
    return (mask_u8 > 0).float()  # (1,H,W) float {0,1}


class JointTransformTensor:
    def __init__(
        self,
        train: bool = True,
        resize: Optional[Tuple[int, int]] = None,
        random_hflip: float = 0.5,
        random_vflip: float = 0.5,
        random_rotate90: bool = True,
        normalize: bool = True,
    ):
        self.train = train
        self.resize = resize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.random_rotate90 = random_rotate90
        self.normalize = normalize

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, img_u8: torch.Tensor, mask_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize first (tensor input supported)
        if self.resize is not None:
            img_u8 = TF.resize(img_u8, self.resize, interpolation=InterpolationMode.BILINEAR)
            mask_u8 = TF.resize(mask_u8, self.resize, interpolation=InterpolationMode.NEAREST)

        if self.train:
            if self.random_hflip > 0 and torch.rand(1).item() < self.random_hflip:
                img_u8 = TF.hflip(img_u8)
                mask_u8 = TF.hflip(mask_u8)

            if self.random_vflip > 0 and torch.rand(1).item() < self.random_vflip:
                img_u8 = TF.vflip(img_u8)
                mask_u8 = TF.vflip(mask_u8)

            if self.random_rotate90:
                k = int(torch.randint(0, 4, (1,)).item())
                if k:
                    img_u8 = torch.rot90(img_u8, k, dims=(1, 2))
                    mask_u8 = torch.rot90(mask_u8, k, dims=(1, 2))

        # Convert image to float in [0,1]
        img_f = img_u8.float() / 255.0
        # Mask to {0,1}
        mask_f = _mask_to_binary_tensor(mask_u8)

        if self.normalize:
            img_f = TF.normalize(img_f, mean=self.mean, std=self.std)

        return img_f, mask_f


def build_transforms(train: bool = True, image_size: Optional[Tuple[int, int]] = None) -> JointTransformTensor:
    return JointTransformTensor(
        train=train,
        resize=image_size,
        random_hflip=0.5 if train else 0.0,
        random_vflip=0.5 if train else 0.0,
        random_rotate90=True if train else False,
        normalize=True,
    )


class WHUDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_meta: bool = False,
        strict_pairing: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        self.images_dir = self.root_dir / self.split / "image"
        self.masks_dir = self.root_dir / self.split / "label"

        self.transform = transform
        self.return_meta = return_meta
        self.strict_pairing = strict_pairing
        self.samples = self._build_index()

    def _build_index(self) -> List[Tuple[Path, Path]]:
        img_files = _list_images(self.images_dir)
        if not img_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

        mask_files = _list_images(self.masks_dir)
        if not mask_files:
            raise RuntimeError(f"No masks found in {self.masks_dir}")

        mask_map: Dict[str, Path] = {_stem(p): p for p in mask_files}

        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []

        for img_path in img_files:
            s = _stem(img_path)
            mask_path = mask_map.get(s)
            if mask_path is None:
                missing.append(s)
                continue
            pairs.append((img_path, mask_path))

        if self.strict_pairing and missing:
            ex = ", ".join(missing[:10])
            raise RuntimeError(f"Missing masks for {len(missing)} images. Examples: {ex}")

        if not pairs:
            raise RuntimeError("No valid image-mask pairs found.")

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, mask_path = self.samples[index]

        img = _load_rgb_tensor(img_path)    # (3,H,W) uint8
        mask = _load_mask_tensor(mask_path) # (1,H,W) uint8

        # If sizes mismatch, align mask to image size
        if img.shape[1:] != mask.shape[1:]:
            mask = TF.resize(mask, size=list(img.shape[1:]), interpolation=InterpolationMode.NEAREST)

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask)
        else:
            img_t = img.float() / 255.0
            img_t = TF.normalize(img_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            mask_t = _mask_to_binary_tensor(mask)

        if self.return_meta:
            meta = {
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "stem": img_path.stem,
                "split": self.split,
                "img_hw": tuple(img.shape[1:]),
                "mask_hw": tuple(mask.shape[1:]),
            }
            return img_t, mask_t, meta

        return img_t, mask_t
