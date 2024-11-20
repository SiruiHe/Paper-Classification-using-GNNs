import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.loader import DataLoader
from tuned_baseline import *
from construct_dataset.create_dataset import CustomArxivDataset
import torch
import gc
import argparse


def objective(trial):
    parser = argparse.ArgumentParser(description="Select model type for training.")
    parser.add_argument("--model", type=str, default="mlp", 
                        choices=["mlp", "gcn", "gat", "graphsage"], 
                        help="Type of model to use: mlp, gcn, gat, or graphsage")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()        
    gc.collect()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pl.seed_everything(42)

    dataset = CustomArxivDataset(root='construct_dataset/')
    data = dataset[0]
    print(f"\nFeature dimensionality: {data.x.shape[1]}")

    data = data.to('cuda:0')
    
    hidden_channels = trial.suggest_int('hidden_channels', 64, 400)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 5e-4, 1e-2, log=True)
    wd = trial.suggest_float('wd', 5e-4, 1e-2)
    gradient_clip_val = trial.suggest_float('gradient_clip_val', 0.1, 2.0)
    if args.model == "gat":
        att_heads = trial.suggest_int('heads', 2, 8)
    else:
        att_heads = 1

    params = {
        'in_channels': data.x.size(1),
        'hidden_channels': hidden_channels,
        'out_channels': dataset.num_classes,
        'learning_rate': learning_rate,
        'wd' : wd,
        'num_layers': num_layers,
        'dropout': dropout
    }
    model_type = args.model
    print(f"Selected model: {model_type}")
    
    if model_type == "gcn":
        model = GCNBaseline(**params).to('cuda:0')
    elif model_type == "gat":
        model = GATBaseline(**params, heads=att_heads).to('cuda:0')   
    elif model_type == "graphsage":
        model = GraphSAGEBaseline(**params).to('cuda:0')   
    elif model_type == "mlp":
        model = MLPBaseline(**params).to('cuda:0')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    callbacks = [
        ModelCheckpoint(monitor='val_acc'),
        EarlyStopping(monitor='val_acc', patience=30, mode = 'max')
    ]
    
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu'if torch.cuda.is_available() else 'cpu',
        devices=[0] if torch.cuda.is_available() else None,
        callbacks=callbacks,
        enable_progress_bar=True,
        precision='16-mixed',
        gradient_clip_val=gradient_clip_val,
    )
    
    train_loader = DataLoader([data],batch_size=64)
    val_loader = DataLoader([data],batch_size=64)
    test_loader = DataLoader([data],batch_size=64)
    
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(model, test_loader)
    test_acc = result[0]['test_acc']
    
    return test_acc
    

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")