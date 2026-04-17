"""
✅ COMPLETE EDUCATIONAL SCRIPT: PyTorch Lightning + MLflow (Local Mac Version)
================================================================================
This script teaches you how to structure a professional training workflow using:
- LightningDataModule → clean data handling
- LightningModule     → model + training logic in one place
- Custom Lightning Callback → full control over MLflow logging
- MLflow → track hyperparameters, metrics, checkpoints, plots & model

Run this as a normal .py file on your Mac Mini M1:
    python train_cifar10_lightning_mlflow.py
"""

import os
import subprocess
import time
import warnings
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Accuracy, ConfusionMatrix

# ===================================================================
# LOCAL HELPER FUNCTIONS
# ===================================================================
def show_ui_navigation_instructions():
    """Clear instructions for the MLflow UI."""
    print("""
================================================================
HOW TO USE THE MLflow UI (Local Version)
================================================================
1. After training, open your browser → http://127.0.0.1:5000
2. Click experiment "CIFAR10_CNN_Experiment" on the left
3. Click the top run (your latest one)
4. Explore tabs: Overview (charts), Parameters, Artifacts
5. In Artifacts you will see:
   • confusion_matrix.png
   • best model checkpoint (.pt)
   • full PyTorch model folder
Press Enter in this terminal when you are done viewing the UI.
""")

import webbrowser   # ← add this import at the top of your script

def start_mlflow_ui(port=5000):
    """Start MLflow UI and automatically open browser cleanly."""
    print("\nStarting MLflow UI...")
    
    process = subprocess.Popen(
        ['mlflow', 'ui', '--host', '127.0.0.1', '--port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(4)  # give it a bit more time to start
    
    url = f"http://127.0.0.1:{port}"
    print(f"✅ MLflow UI is running at → {url}")
    
    # Auto-open in default browser (clean new tab)
    webbrowser.open(url)
    print("🔄 Browser should open automatically now...")
    
    print("\n" + "="*70)
    print("🌐 MLflow UI is LIVE!")
    print("="*70)
    print("You can now explore your experiment (parameters, metrics, artifacts, etc.)")
    print("When you are finished, press Enter in this terminal to stop the server.")
    print("="*70)
    
    return process

# ===================================================================
# GLOBAL SETTINGS & REPRODUCIBILITY
# ===================================================================
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ===================================================================
# 1. DATA MODULE
# ===================================================================
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data/CIFAR10_data', batch_size=128, num_workers=1):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        if self.data_dir.exists():
            print("✅ CIFAR-10 data already downloaded.")
        else:
            print("⬇️ Downloading CIFAR-10...")
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.cifar_train = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform_train)
        self.cifar_val   = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_val, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.num_workers, pin_memory=True)

# ===================================================================
# 2. MODEL
# ===================================================================
class SimpleCNN(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ===================================================================
# 3. CONFUSION MATRIX + CUSTOM MLflow CALLBACK
# ===================================================================
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Final Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    filename = 'confusion_matrix.png'
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename

class MLflowLoggingCallback(Callback):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.best_accuracy = 0.0
        self.model_save_dir = "./best_model"
        os.makedirs(self.model_save_dir, exist_ok=True)

    def on_train_start(self, trainer, pl_module):
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("initial_lr", pl_module.hparams.learning_rate)
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        mlflow.log_param("batch_size", trainer.datamodule.batch_size)
        mlflow.log_param("random_seed", RANDOM_SEED)
        print("✅ Hyperparameters logged to MLflow.")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if "val_acc" in metrics:
            acc = metrics["val_acc"].item() * 100
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                checkpoint = {'epoch': epoch+1, 'model_state_dict': pl_module.state_dict(),
                              'val_loss': metrics.get("val_loss", torch.tensor(float('inf'))).item(),
                              'accuracy': acc, 'random_seed': RANDOM_SEED}
                path = os.path.join(self.model_save_dir, f'best_model_epoch_{epoch+1}.pt')
                torch.save(checkpoint, path)
                mlflow.log_artifact(path, artifact_path="checkpoints")
                print(f"🏆 New best model! Accuracy = {acc:.2f}%")

    def on_train_end(self, trainer, pl_module):
        print("\n📊 Generating final confusion matrix...")
        confmat = ConfusionMatrix(task="multiclass", num_classes=10).to(pl_module.device)
        pl_module.eval()
        with torch.no_grad():
            for batch in trainer.val_dataloaders:
                x, y = [t.to(pl_module.device) for t in batch]
                preds = torch.argmax(pl_module(x), dim=1)
                confmat.update(preds, y)
        cm = confmat.compute().cpu().numpy()
        cm_path = plot_confusion_matrix(cm, self.classes)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        example = next(iter(trainer.val_dataloaders))[0].cpu().numpy()
        pl_module.to("cpu")
        mlflow.pytorch.log_model(pl_module, "cifar10_cnn_model_final", input_example=example)
        mlflow.log_metric("best_accuracy", self.best_accuracy)
        print(f"🎉 Training finished! Best accuracy: {self.best_accuracy:.2f}%")

# ===================================================================
# 4. MAIN – TRAINING + MLflow UI THAT STAYS OPEN
# ===================================================================
if __name__ == "__main__":
    print("🚀 Starting Lightning + MLflow CIFAR-10 training...\n")

    data_module = CIFAR10DataModule(batch_size=128, num_workers=1)
    mlflow.set_experiment("CIFAR10_CNN_Experiment")

    with mlflow.start_run(run_name="SimpleCNN_Run_001") as run:
        model = SimpleCNN(learning_rate=0.001)
        callback = MLflowLoggingCallback(classes=data_module.classes)

        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices=1,
            logger=False,
            callbacks=[callback],
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=False,
        )

        trainer.fit(model, data_module)

        print(f"\n✅ Training complete! MLflow Run ID = {run.info.run_id}")

    # ====================== MLflow UI THAT STAYS OPEN ======================
    show_ui_navigation_instructions()
    mlflow_process = start_mlflow_ui()

    print("\n🌐 MLflow UI is now LIVE at: http://127.0.0.1:5000")
    print("You can now open the link in your browser and explore everything.")

    # Keep the script alive so the UI doesn't die
    input("\n✅ Press Enter when you are finished viewing the MLflow UI to stop the server...")

    mlflow_process.terminate()
    print("MLflow UI stopped. Have a great day! 🚀")