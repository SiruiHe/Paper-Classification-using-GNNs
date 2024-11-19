import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from my_models import *
from construct_dataset.create_dataset import CustomArxivDataset
from torch_geometric.loader import DataLoader
import json
import os
from datetime import datetime

class CustomProgressBar(pl.callbacks.ProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = self.get_metrics(trainer, pl_module)
        epoch = trainer.current_epoch
        loss = pl_module.current_loss
        train_acc = trainer.callback_metrics.get("train_acc", 0)
        val_acc = trainer.callback_metrics.get("val_acc", 0)
        test_acc = trainer.callback_metrics.get("test_acc", 0)
        best_val = pl_module.best_val_acc
        best_test = pl_module.best_test_acc
        
        print(f"Epoch: {epoch:3d}, Loss: {loss:.4f}, Train: {train_acc:.2%}, "
              f"Valid: {val_acc:.2%}, Test: {test_acc:.2%}, "
              f"Best Valid: {best_val:.2%}, Best Test: {best_test:.2%}")


def create_model(model_type, data, dataset, preprocess_mode='basic'):
    """Create model instance based on model type"""
    model_params = {}

    if model_type == "mlp":
        model_params = {
            'in_channels': data.x.size(1),
            'hidden_channels': 256,
            'out_channels': dataset.num_classes,
            'num_layers': 3,
            'dropout': 0.5
        }
        return MLPBaseline(**model_params), model_params
        
    elif model_type == "gcn":
        model_params = {
            'in_channels': data.x.size(1),
            'hidden_channels': 256,
            'out_channels': dataset.num_classes,
            'num_layers': 3,
            'dropout': 0.5,
            'preprocess_mode': preprocess_mode
        }
        return GCN(**model_params), model_params
        
    elif model_type == "gat":
        model_params = {
            'in_channels': data.x.size(1),
            'hidden_channels': 32,
            'out_channels': dataset.num_classes,
            'num_layers': 3,
            'dropout': 0.5,
            'heads': 8, # If OOM, change to 4
            'preprocess_mode': preprocess_mode
        }
        return GAT(**model_params), model_params
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")



def train_and_evaluate(model, train_loader, val_loader, test_loader, model_type, preprocess_mode=None):
    """Helper function to train and evaluate a model with specific configuration"""
    
    # Create unique experiment name
    experiment_name = f"{model_type}"
    if preprocess_mode:
        experiment_name += f"_{preprocess_mode}"
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc',
            dirpath='checkpoints',
            filename=f'{experiment_name}-{{epoch:02d}}-{{val_acc:.2f}}',
            save_top_k=1,  
            save_last=False,
            mode='max',
            every_n_epochs=1, 
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max'
        ),
        CustomProgressBar()  # Add custom progress bar
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and clean up other checkpoints
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nLoading best model from {best_model_path}")
        best_model = model.load_from_checkpoint(best_model_path, **model.hparams)
        
        # clean checkpoints
        checkpoint_dir = 'checkpoints'
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            if file_path != best_model_path:  # save best model
                os.remove(file_path)
        
        results = trainer.test(best_model, test_loader)
    else:
        results = trainer.test(model, test_loader)
    
    return results[0]

def main(model_type='gcn', compare_preprocess=True):
    """Main training function with support for model selection and preprocessing comparison"""
    
    # Load dataset
    dataset = CustomArxivDataset(root='construct_dataset/')
    data = dataset[0]
    
    # Print dataset statistics
    print(f'Dataset Statistics:')
    print(f'Number of nodes: {data.x.size(0)}')
    print(f'Number of edges: {data.edge_index.size(1)}')
    print(f"Selected model: {model_type}")
    
    # Create data loaders
    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    test_loader = DataLoader([data], batch_size=1)
    
    # Store results for comparison
    all_results = {}
    model_configs = {}
    
    # Handle preprocessing comparison for supported models
    if model_type != "mlp" and compare_preprocess:
        for preprocess_mode in ['basic', 'arxiv']:
            print(f"\nTraining {model_type.upper()} with {preprocess_mode} preprocessing...")
            model, params = create_model(model_type, data, dataset, preprocess_mode)
            results = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                model_type, preprocess_mode
            )
            all_results[f"{model_type}_{preprocess_mode}"] = results
            model_configs[exp_name] = params
    else:
        model, params = create_model(model_type, data, dataset)
        results = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            model_type
        )
        all_results[model_type] = results
        model_configs[model_type] = params
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/results_{model_type}_{timestamp}.json"
    results_content = {
        "model_type": model_type,
        "model_configs": model_configs,
        "results": all_results
    }
    with open(results_filename, 'w') as f:
        json.dump(results_content, f, indent=4)
    
    # Print comparison if multiple experiments were run
    if len(all_results) > 1:
        print("\nResults Comparison:")
        for exp_name, metrics in all_results.items():
            print(f"\n{exp_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

if __name__ == '__main__':
    # Example usage:
    # main(model_type='gcn', compare_preprocess=True)  # Compare preprocessing for GCN
    main(model_type='gat', compare_preprocess=False)  # Compare preprocessing for GAT
    # main(model_type='mlp', compare_preprocess=False)  # Run MLP baseline