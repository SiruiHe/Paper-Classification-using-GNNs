import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from my_models import *
from dataset.create_dataset import CustomArxivDataset
from torch_geometric.loader import DataLoader
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

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

def plot_accuracy_curves(train_accs, val_accs, model_type, preprocess_mode=None):
    """Plot training and validation accuracy curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curves for {model_type.upper()}' + 
             (f' ({preprocess_mode} preprocessing)' if preprocess_mode else ''))
    plt.legend()
    plt.grid(True)
    
    # Create accuracy directory if it doesn't exist
    os.makedirs('accuracy', exist_ok=True)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'accuracy/{model_type}_{preprocess_mode}_{timestamp}.png'
    plt.savefig(filename)
    plt.close()
    print(f"\nAccuracy curves saved to {filename}")
    
def plot_loss_curves(train_losses, val_losses, model_type, preprocess_mode=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for {model_type.upper()}' + 
             (f' ({preprocess_mode} preprocessing)' if preprocess_mode else ''))
    plt.legend()
    plt.grid(True)
    
    # Create loss directory if it doesn't exist
    os.makedirs('loss', exist_ok=True)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'loss/{model_type}_{preprocess_mode}_{timestamp}.png'
    plt.savefig(filename)
    plt.close()
    print(f"\nLoss curves saved to {filename}")
    
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
            'hidden_channels': 64,
            'out_channels': dataset.num_classes,
            'num_layers': 3,
            'dropout': 0.5,
            'heads': 2, 
            'preprocess_mode': preprocess_mode
        }
        return GAT(**model_params), model_params
    
    elif model_type == "gat_dgl":
        model_params = {
            'in_channels': data.x.size(1),
            'hidden_channels': 64,
            'out_channels': dataset.num_classes,
            'num_layers': 3,
            'dropout': 0.75,
            'heads': 4,
            'attn_drop': 0.05,
            'norm': 'both',
            'preprocess_mode': preprocess_mode
        }
        return GATAdapter(**model_params), model_params
    
    elif model_type == "sage":
        model_params = {
            'in_channels': data.x.size(1),
            'hidden_channels': 256,
            'out_channels': dataset.num_classes,
            'num_layers': 3, 
            'dropout': 0.5,
            'preprocess_mode': preprocess_mode
        }
        return GraphSAGE(**model_params), model_params
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")



def train_and_evaluate(model, train_loader, val_loader, test_loader, model_type, preprocess_mode=None):
    """Helper function to train and evaluate a model with specific configuration"""
    
    warnings.filterwarnings('ignore')
    
    print("\nDetailed Model Summary:")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'Layer':<40} {'Output Shape':<20} {'Param #':<10}")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"{name:<40} {str(list(param.shape)):<20} {param.numel():<10}")
    
    print("-" * 80)
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print("-" * 80)
    
    # Create unique experiment name
    experiment_name = f"{model_type}"
    if preprocess_mode:
        experiment_name += f"_{preprocess_mode}"
        
    class LossTracker(pl.Callback):
        def __init__(self):
            super().__init__()
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
            self.best_epoch = 0
            self.best_val_acc = 0
                
        def on_train_epoch_end(self, trainer, pl_module):
            # Record losses at the end of each epoch
            train_loss = float(pl_module.current_loss)
            val_loss = float(trainer.callback_metrics.get("val_loss", 0))
            train_acc = float(trainer.callback_metrics.get("train_acc", 0))
            val_acc = float(trainer.callback_metrics.get("val_acc", 0))
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Track best validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = trainer.current_epoch
        
        def on_train_end(self, trainer, pl_module):
            # Plot curves at the end of training
            if len(self.train_losses) > 0:
                plot_loss_curves(self.train_losses, self.val_losses, 
                            model_type, preprocess_mode)
                plot_accuracy_curves(self.train_accs, self.val_accs,
                                model_type, preprocess_mode)
                
                # Print best metrics
                print(f"\nBest Validation Metrics (Epoch {self.best_epoch + 1}):")
                print(f"  Train Loss: {self.train_losses[self.best_epoch]:.4f}")
                print(f"  Valid Loss: {self.val_losses[self.best_epoch]:.4f}")
                print(f"  Train Accuracy: {self.train_accs[self.best_epoch]:.2%}")
                print(f"  Valid Accuracy: {self.val_accs[self.best_epoch]:.2%}")
    
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
            patience=50,
            mode='max'
        ),
        CustomProgressBar(),
        LossTracker()  # Add loss tracker
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        # logger=False
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and clean up other checkpoints
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nLoading best model from {best_model_path}")
        model_class = type(model)
        best_model = model_class.load_from_checkpoint(best_model_path)
        
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

def train_gnn_model(model_type='gcn', preprocess_mode='compare', embedding_type='scibert'):
    """Main training function with support for model selection and preprocessing comparison"""
    
    # Load dataset
    dataset = CustomArxivDataset(root='dataset/', embedding_type=embedding_type)
    data = dataset[0]
    
    # Print dataset statistics
    print(f'Dataset Statistics:')
    print(f'Number of nodes: {data.x.size(0)}')
    print(f'Number of edges: {data.edge_index.size(1)}')
    print(f"Selected model: {model_type}")
    print(f"Using embeddings: {embedding_type}")
    
    # Create data loaders
    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    test_loader = DataLoader([data], batch_size=1)
    
    # Store results for comparison
    all_results = {}
    model_configs = {}
    
    # Handle different preprocessing modes
    if model_type == "mlp":
        # MLP doesn't use preprocessing
        exp_name = model_type
        model, params = create_model(model_type, data, dataset)
        results = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            model_type
        )
        all_results[exp_name] = results
        model_configs[exp_name] = params
    elif preprocess_mode == 'compare':
        # Compare both preprocessing methods
        for curr_preprocess in ['basic', 'arxiv']:
            exp_name = f"{model_type}_{curr_preprocess}"
            print(f"\nTraining {model_type.upper()} with {curr_preprocess} preprocessing...")
            model, params = create_model(model_type, data, dataset, curr_preprocess)
            results = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                model_type, curr_preprocess
            )
            all_results[exp_name] = results
            model_configs[exp_name] = params
    else:
        # Use specified preprocessing mode
        exp_name = f"{model_type}_{preprocess_mode}"
        print(f"\nTraining {model_type.upper()} with {preprocess_mode} preprocessing...")
        model, params = create_model(model_type, data, dataset, preprocess_mode)
        results = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            model_type, preprocess_mode
        )
        all_results[exp_name] = results
        model_configs[exp_name] = params
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/results_{model_type}_{timestamp}.json"
    results_content = {
        "model_type": model_type,
        "preprocess_mode": preprocess_mode,
        "embedding_type": embedding_type,
        "model_configs": model_configs,
        "results": all_results
    }
    with open(results_filename, 'w') as f:
        json.dump(results_content, f, indent=4)
    
    # Print final results for all experiments
    print(f"\nFinal Results (Model: {model_type.upper()}, Embedding: {embedding_type}):")
    for exp_name, metrics in all_results.items():
        test_acc = metrics.get('test_acc', 0)
        print(f"  {exp_name}: Test Accuracy = {test_acc:.2%}")
    
    return all_results

if __name__ == '__main__':
    # Example usage:
    # train_gnn_model(model_type='mlp', embedding_type='word2vec')  # Run MLP baseline with word2vec
    # train_gnn_model(model_type='gcn', embedding_type='scibert')  # Run GCN with scibert
    
    # train_gnn_model(model_type='gcn', preprocess_mode='compare')  # Compare preprocessing methods
    # train_gnn_model(model_type='gat', preprocess_mode='basic')  # Use basic preprocessing
    # train_gnn_model(model_type='sage', preprocess_mode='arxiv')  # Use arxiv preprocessing
    
    train_gnn_model(model_type='gcn', preprocess_mode='basic', embedding_type='word2vec')
    