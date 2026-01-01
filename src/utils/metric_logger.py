import csv
from pathlib import Path

class MetricLogger:
    def __init__(self, filepath:str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            with open(self.filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'epoch',
                    'train_loss',
                    'val_loss',
                    'val_dice',
                    'val_iou',
                    'val_accuracy'
                    ])
    
    def log(self, epoch:int, train_loss:float, val_loss:float, val_dice:float, val_iou:float, acc:float):
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_dice:.4f}",
                f"{val_iou:.4f}",
                f"{acc:.4f}"
            ])
        