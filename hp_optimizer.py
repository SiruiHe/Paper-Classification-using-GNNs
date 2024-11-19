import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader
from GNN_Project.my_models import GCNBaseline
from construct_dataset.create_dataset import CustomArxivDataset
import torch
import multiprocessing as mp
import os
import nvidia_smi
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

def print_gpu_utilization():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def objective(trial):
    try:
        torch.set_float32_matmul_precision('medium')
        torch.cuda.empty_cache()        
        gc.collect()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        # Check CUDA availability and print device info
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print("\nInitial GPU state:")
        print_gpu_utilization()

        # Set random seed
        pl.seed_everything(42)
        
        # Calculate optimal workers
        num_workers = min(mp.cpu_count(), 19)
        
        # Load dataset
        dataset = CustomArxivDataset(root='construct_dataset/')
        data = dataset[0]
        print(f"\nFeature dimensionality: {data.x.shape[1]}")

        data = data.to('cuda:0')
        
        # Hyperparameters to optimize
        hidden_channels = trial.suggest_int('hidden_channels', 128, 200)
        num_layers = trial.suggest_int('num_layers', 2, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.02, log=True)
        #patience = trial.suggest_int('patience', 10, 40)
        #monitor_metric = trial.suggest_categorical('monitor_metric', ['val_loss', 'val_acc'])
        #max_epochs = trial.suggest_int('max_epochs', 150, 300)
        gradient_clip_val = trial.suggest_float('gradient_clip_val', 0.1, 1.0)
        accumulate_grad_batches = trial.suggest_int('accumulate_grad_batches', 1, 4)
        # Create model
        model = GCNBaseline(
            in_channels=data.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            learning_rate=learning_rate,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        model.gradient_checkpointing = True

        # Configure callbacks
        callbacks = [
            ModelCheckpoint(monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=15, mode='min')
        ]
        
        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=500,
            accelerator='gpu',
            devices=[0],
            callbacks=callbacks,
            precision='16-mixed', 
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            inference_mode=False 
        )
        
        # DataLoaders with workers
        train_loader = DataLoader(
            [data],
            batch_size=1,
        )
        val_loader = DataLoader(
            [data],
            batch_size=1,
        )
        
        # Train the model
        trainer.fit(model, train_loader, val_loader)
        
        # Evaluate the model
        result = trainer.test(model, val_loader)
        test_acc = result[0]['test_acc']
        
        return test_acc
    
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        print("\nFinal GPU state:")
        print_gpu_utilization()
        raise

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")