from pathlib import Path
from PIL import Image

def convert_dir(split: str):
    img_dir = Path("data/raw") / split / "image"
    lab_dir = Path("data/raw") / split / "label"

    out_img = Path("data") / split / "image"
    out_lab = Path("data") / split / "label"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    # Images
    for p in img_dir.glob("*.tif*"):
        im = Image.open(p).convert("RGB")
        im.save(out_img / (p.stem + ".png"), optimize=True)

    # Labels (keep binary)
    for p in lab_dir.glob("*.tif*"):
        m = Image.open(p).convert("L")
        m.save(out_lab / (p.stem + ".png"), optimize=True)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_dir(split)
    print("Done. New dataset at: data/")
