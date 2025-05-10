import torch
import os
import sys
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch import nn
import sys  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
from timeit import default_timer as timer 
import matplotlib.pyplot as plt
import json
import itertools
import pandas as pd
from datetime import datetime as dt
sys.path.append("..")
from data_loader_java import *
from data_loader_c import *
from data_loader_cpp import *

import torch.utils.data

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        return None

def get_default_hyperparameters():
    hyperparams = {
        "BATCH_SIZE": 10,
        "NUM_WORKERS": 0,
        "NUM_EPOCHS": 50,
        "LR": 0.001,
        "INPUT_SIZE": 22,
        "HIDDEN_SIZES": [50, 10, 1],
        "LOSS_FN": nn.BCELoss(),
        "OPTIMIZER": torch.optim.Adam,
        "SCHEDULER": {
            "type": torch.optim.lr_scheduler.StepLR,
            "params": {"step_size": 10, "gamma": 1}
        },
        "DROPOUT": 0.5
    }
    hyperparams['PATIENCE'] = int(hyperparams['NUM_EPOCHS'] * 0.2)
    return hyperparams

def load_commit_history_data(language="CPP", batch_size=10, num_workers=0, split=0.7):

    label_column = "is_abandoned"
    if language == "Java":
        commit_history_dataset_train = Commit_History_Java_Dataset(label_column, split, "train")
        commit_history_dataset_test = Commit_History_Java_Dataset(label_column, split, "test")
    elif language == "C":
        commit_history_dataset_train = Commit_History_C_Dataset(label_column, split, "train")
        commit_history_dataset_test = Commit_History_C_Dataset(label_column, split, "test")
    elif language == "CPP":
        commit_history_dataset_train = Commit_History_CPP_Dataset(label_column, split, "train")
        commit_history_dataset_test = Commit_History_CPP_Dataset(label_column, split, "test")
    else:
        raise ValueError(f"Unsupported language: {language}")


    all_labels_train = []
    for i in range(len(commit_history_dataset_train)):
        _, label = commit_history_dataset_train[i]
        all_labels_train.append(label.item())
    
    all_labels_train = np.array(all_labels_train)
    class_counts_train = np.bincount(all_labels_train.astype(int))
    if len(class_counts_train) == 0:
         raise ValueError("Training dataset has no labels.")
    elif len(class_counts_train) == 1:
         present_class = class_counts_train.argmax()
         class_counts_train = np.array([class_counts_train[0], 1]) if present_class == 0 else np.array([1, class_counts_train[0]])
         
    class_weights_train = 1.0 / np.maximum(class_counts_train, 1) 
    sample_weights_train = class_weights_train[all_labels_train.astype(int)]
    
    sampler_train = WeightedRandomSampler(
        weights=sample_weights_train,
        num_samples=len(sample_weights_train),
        replacement=True
    )
    
    if len(commit_history_dataset_test) > 0:
        all_labels_test = []
        for i in range(len(commit_history_dataset_test)):
            _, label = commit_history_dataset_test[i]
            all_labels_test.append(label.item())
        all_labels_test = np.array(all_labels_test)
        class_counts_test = np.bincount(all_labels_test.astype(int))
        if len(class_counts_test) < 2: class_counts_test = np.append(class_counts_test, 0) if len(class_counts_test) == 1 and class_counts_test.argmax()==0 else ([0] + class_counts_test.tolist() if len(class_counts_test) == 1 else [0, 0])

    train_dataloader = DataLoader(
        dataset=commit_history_dataset_train, 
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler_train,
        drop_last=True,
        collate_fn=collate_fn_skip_none 
    )
    
    test_dataloader = DataLoader(
        dataset=commit_history_dataset_test, 
        batch_size=batch_size, 
        num_workers=num_workers,
        sampler=None, 
        shuffle=False, 
        drop_last=False,
        collate_fn=collate_fn_skip_none 
    ) if commit_history_dataset_test and len(commit_history_dataset_test) > 0 else None
    
    return train_dataloader, test_dataloader 

class RelNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.5):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):   
        return self.network(x)

def create_model(hyperparams, device="cuda"):

    model = RelNN(
        input_size=hyperparams['INPUT_SIZE'],
        hidden_sizes=hyperparams['HIDDEN_SIZES'],
        dropout=hyperparams['DROPOUT']
    ).to(device)
    
    if hyperparams['OPTIMIZER'] == torch.optim.Adam:
        optimizer = hyperparams['OPTIMIZER'](
            params=model.parameters(), 
            lr=hyperparams['LR'],
            betas=hyperparams['BETAS']
        )
    elif hyperparams['OPTIMIZER'] == torch.optim.SGD:
        optimizer = hyperparams['OPTIMIZER'](
            params=model.parameters(), 
            lr=hyperparams['LR'],
            momentum=hyperparams['MOMENTUM']
        )
    
    scheduler = hyperparams['SCHEDULER']['type'](
        optimizer, 
        **hyperparams['SCHEDULER']['params']
    )
    
    return model, optimizer, scheduler

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device):
    
    model.train()
    train_loss  = 0
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        X, y = batch 
        X, y = X.to(device), y.unsqueeze(dim=1).to(device)
 
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        all_predictions.append(pred.cpu().detach().numpy())
        all_targets.append(y.cpu().numpy())

    predictions = np.concatenate(all_predictions)  
    targets = np.concatenate(all_targets)

    binary_predictions = (predictions >= 0.5).astype(np.float32)

    predictions = predictions.flatten()
    targets = targets.flatten()
    binary_predictions = binary_predictions.flatten()

    metrics = {
        'loss': train_loss / len(dataloader),
        'accuracy': accuracy_score(targets, binary_predictions),
        'precision': precision_score(targets, binary_predictions,zero_division=0),
        'recall': recall_score(targets, binary_predictions,zero_division=0),
        'f1': f1_score(targets, binary_predictions,zero_division=0)
    }

    return metrics

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []

    for batch in dataloader:

        X, y = batch
        X, y = X.to(device), y.unsqueeze(dim=1).to(device)

        with torch.inference_mode():
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            all_predictions.append(pred.cpu().detach().numpy())
            all_targets.append(y.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    binary_predictions = (predictions >= 0.5).astype(np.float32)

    predictions = predictions.flatten()
    targets = targets.flatten()
    binary_predictions = binary_predictions.flatten()
    
    metrics = {
        'loss': test_loss / len(dataloader),
        'accuracy': accuracy_score(targets, binary_predictions),
        'precision': precision_score(targets, binary_predictions,zero_division=0), 
        'recall': recall_score(targets, binary_predictions,zero_division=0),
        'f1': f1_score(targets, binary_predictions,zero_division=0)
    }

    return metrics

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, monitor='test_loss'): 

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.loss_min = float('inf')
        self.monitor = monitor


    def __call__(self, current_loss, model, optimizer, scheduler, epoch, save_path, hyperparameters=None):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(current_loss, model, optimizer, scheduler, epoch, save_path, hyperparameters)
        elif current_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.save_checkpoint(current_loss, model, optimizer, scheduler, epoch, save_path, hyperparameters)
            self.counter = 0
            
    def save_checkpoint(self, current_loss, model, optimizer, scheduler, epoch, save_path, hyperparameters=None):
    
        save_model_state(model, optimizer, scheduler, epoch, save_path, hyperparameters)
        self.loss_min = current_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, 
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          device: torch.device,
          epochs: int,
          writer: torch.utils.tensorboard.SummaryWriter,
          log_dir: str,
          hyperparameters: dict,
          save_interval: int = 10,
          patience: int = 7):
    
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 
        'train_recall': [], 'train_f1': [],
        'test_loss': [], 'test_accuracy': [], 'test_precision': [], 
        'test_recall': [], 'test_f1': [], 'test_roc_auc': [] 
    }
    
    model_artifacts_dir = os.path.join(log_dir, 'model_artifacts')
    os.makedirs(model_artifacts_dir, exist_ok=True)

    save_hyperparameters(hyperparameters, model_artifacts_dir, model)
    
    sample_input = torch.randn(1, hyperparameters['INPUT_SIZE']).to(device)
    

    early_stopping = EarlyStopping(patience=patience, verbose=True, monitor='test_loss') 
    
    empty_metrics = {'loss': float('nan'), 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    for epoch in tqdm(range(epochs)):
        train_metrics = train_step(model=model,
                                 dataloader=train_dataloader,
                                 loss_fn=loss_fn,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 device=device)
        

        if test_dataloader:
            test_metrics = test_step(model=model,
                                   dataloader=test_dataloader,
                                   loss_fn=loss_fn,
                                   device=device)
        else:
             test_metrics = empty_metrics 

        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        history['test_precision'].append(test_metrics['precision'])
        history['test_recall'].append(test_metrics['recall'])
        history['test_f1'].append(test_metrics['f1'])

        writer.add_scalar("train_loss", train_metrics['loss'], epoch)
        writer.add_scalar("train_accuracy", train_metrics['accuracy'], epoch)
        writer.add_scalar("train_f1", train_metrics['f1'], epoch)
        writer.add_scalar("train_precision", train_metrics['precision'], epoch)
        writer.add_scalar("train_recall", train_metrics['recall'], epoch)

        writer.add_scalar("test_loss", test_metrics['loss'], epoch)
        writer.add_scalar("test_accuracy", test_metrics['accuracy'], epoch)
        writer.add_scalar("test_f1", test_metrics['f1'], epoch)
        writer.add_scalar("test_precision", test_metrics['precision'], epoch)
        writer.add_scalar("test_recall", test_metrics['recall'], epoch)
        
        loss_to_monitor = test_metrics['loss'] if early_stopping.monitor == 'test_loss' else train_metrics['loss']
        if not np.isnan(loss_to_monitor): 
             early_stopping(loss_to_monitor, model, optimizer, scheduler, epoch, model_artifacts_dir, hyperparameters)
             if early_stopping.early_stop:
                 break
        else:
             print(f"Warning: Monitored loss ({early_stopping.monitor}) is NaN at epoch {epoch}. Skipping early stopping check.")

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1 or early_stopping.early_stop: 
             save_model_state(model, optimizer, scheduler, epoch, model_artifacts_dir, hyperparameters)

        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}")
            print(f"  Train      - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            if test_dataloader: 
                print(f"  Test       - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    if history['train_loss']:
        final_metrics_text = (
            f"# Final Model Performance (Epoch {epoch})\n\n"
            f"## Training Set\n"
            f"* **Loss**: {history['train_loss'][-1]:.4f}\n"
            f"* **Accuracy**: {history['train_accuracy'][-1]:.4f}\n"
            f"* **Precision**: {history['train_precision'][-1]:.4f}\n"
            f"* **Recall**: {history['train_recall'][-1]:.4f}\n"
            f"* **F1 Score**: {history['train_f1'][-1]:.4f}\n\n"
            f"## Test Set\n"
            f"* **Loss**: {history['test_loss'][-1]:.4f}\n"
            f"* **Accuracy**: {history['test_accuracy'][-1]:.4f}\n"
            f"* **Precision**: {history['test_precision'][-1]:.4f}\n"
            f"* **Recall**: {history['test_recall'][-1]:.4f}\n"
            f"* **F1 Score**: {history['test_f1'][-1]:.4f}\n"
        )
    else:
        final_metrics_text = "# Final Model Performance\n\n No training history recorded."
    writer.add_text("final_metrics", final_metrics_text)
        
    return history

def start_training(writer, log_dir, hyperparameters=None, language="CPP"):
 
    if hyperparameters is None:
        hyperparameters = get_default_hyperparameters()
    
    train_dataloader, test_dataloader = load_commit_history_data(
        language=language,
        batch_size=hyperparameters['BATCH_SIZE'],
        num_workers=hyperparameters['NUM_WORKERS']
    )
         
    model, optimizer, scheduler = create_model(hyperparameters, device)
    
    start_time = timer()
    
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=hyperparameters['LOSS_FN'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=hyperparameters['NUM_EPOCHS'],
        writer=writer,
        log_dir=log_dir,
        hyperparameters=hyperparameters,
        patience=hyperparameters['PATIENCE']
    )
    
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    with open(os.path.join(log_dir, 'model_artifacts', 'training_results.json'), 'w') as f:
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)) and all(isinstance(v, (int, float, np.number)) for v in value):
                serializable_results[key] = [float(v) for v in value]
            else:
                 serializable_results[key] = value 
        json.dump(serializable_results, f, indent=4)

    model.eval()


    def get_predictions_actuals(dataloader, model, device):
        all_preds = []
        all_actuals = []
        if dataloader is None:
            return np.array([]), np.array([]) 
            
        model.eval()
        with torch.inference_mode():
            for batch in dataloader: 
                 if batch is None: continue
                 if not batch: continue
                 X, y = batch
                 X, y = X.to(device), y.to(device)
                 preds = model(X)
                 all_preds.append(preds.cpu().numpy())
                 all_actuals.append(y.cpu().numpy())
        
        if not all_preds:
            return np.array([]), np.array([])

        predictions_proba = np.concatenate(all_preds).flatten()
        actuals = np.concatenate(all_actuals).flatten().astype(np.int32)
        predictions_binary = (predictions_proba >= 0.5).astype(np.int32)
        return predictions_binary, actuals


    if test_dataloader:
        try:
            preds_test, actuals_test = get_predictions_actuals(test_dataloader, model, device)
            if len(actuals_test) > 0:
                cm_test = confusion_matrix(actuals_test, preds_test)
    

                plt.figure(figsize=(8, 6))
                disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Not Abandoned', 'Abandoned'])
                disp_test.plot(cmap=plt.cm.Blues, values_format='d')
                plt.title('Test Set Confusion Matrix')
                plt.savefig(os.path.join(log_dir, 'model_artifacts', 'confusion_matrix_test.png'), bbox_inches='tight', dpi=300)
                plt.close() 
            else:
                 print("Info: No test predictions generated for confusion matrix.")
        except Exception as e:
            print(f"Could not generate test confusion matrix: {e}")
    else:
         print("Info: Skipping test confusion matrix as test dataloader is None.")


    writer.flush()
    return model, results



def save_hyperparameters(hyperparams, save_path, model=None):

    params_to_save = hyperparams.copy()
    
    serializable_params = {}
    for key, value in params_to_save.items():
        if isinstance(value, type) or callable(value):
            serializable_params[key] = str(value)
        elif key == 'LOSS_FN':
            serializable_params[key] = value.__class__.__name__
        elif key == 'HIDDEN_SIZES':
            serializable_params[key] = value
        elif key == 'SCHEDULER':
            scheduler_dict = {
                'type': value['type'].__name__,
                'params': value['params']
            }
            serializable_params[key] = scheduler_dict
        else:
            serializable_params[key] = value
    
    if model is not None:
        serializable_params['MODEL_ARCHITECTURE'] = str(model)
    
    hp_path = os.path.join(save_path, "hyperparameters.json")
    
    os.makedirs(os.path.dirname(hp_path), exist_ok=True)
    
    with open(hp_path, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    return hp_path


def save_model_state(model, optimizer, scheduler, epoch, save_path, hyperparameters=None):

    checkpoint_path = os.path.join(save_path, f"model_checkpoint_epoch_{epoch}.pt")
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if hyperparameters and epoch == hyperparameters.get('NUM_EPOCHS', 0) - 1:
        final_model_path = os.path.join(save_path, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
    
    return checkpoint_path



class EnhancedCustomWriter:
    @staticmethod
    def create_Writer(hyperparameters, experiment_name, model_name, dataset_name, extra=None):

        timestamp = dt.now().strftime("%Y%m%d_%H%M")
        
        if extra:
            log_dir = f"runs/{experiment_name}/{model_name}/{dataset_name}/{timestamp}_{extra}"
        else:
            log_dir = f"runs/{experiment_name}/{model_name}/{dataset_name}/{timestamp}"
        
        hidden_sizes_str = "_".join(map(str, hyperparameters['HIDDEN_SIZES']))
        
        hyperparams_text = "# Model Hyperparameters\n\n"
        hyperparams_text += f"## Architecture\n"
        hyperparams_text += f"* **Model Type**: {model_name}\n"
        hyperparams_text += f"* **Input Size**: {hyperparameters['INPUT_SIZE']}\n"
        hyperparams_text += f"* **Hidden Sizes**: {hyperparameters['HIDDEN_SIZES']}\n"
        hyperparams_text += f"* **Hidden Layer Count**: {len(hyperparameters['HIDDEN_SIZES'])}\n"
        hyperparams_text += f"* **Dropout Rate**: {hyperparameters['DROPOUT']}\n\n"
        
        hyperparams_text += f"## Training Configuration\n"
        hyperparams_text += f"* **Learning Rate**: {hyperparameters['LR']}\n"
        hyperparams_text += f"* **Batch Size**: {hyperparameters['BATCH_SIZE']}\n"
        hyperparams_text += f"* **Epochs**: {hyperparameters['NUM_EPOCHS']}\n"
        hyperparams_text += f"* **Optimizer**: {hyperparameters['OPTIMIZER'].__name__}\n"
        hyperparams_text += f"* **Loss Function**: {hyperparameters['LOSS_FN'].__class__.__name__}\n"
        
        scheduler_type = hyperparameters['SCHEDULER']['type'].__name__
        hyperparams_text += f"* **Learning Rate Scheduler**: {scheduler_type}\n"
        for param_name, param_value in hyperparameters['SCHEDULER']['params'].items():
            hyperparams_text += f"  * **{param_name}**: {param_value}\n"
        
        hyperparams_text += f"\n## Dataset\n"
        hyperparams_text += f"* **Language**: {dataset_name}\n"
        hyperparams_text += f"* **Workers**: {hyperparameters['NUM_WORKERS']}\n"
        

        writer = SummaryWriter(
            log_dir=log_dir,
            comment='',
            purge_step=None,
            max_queue=1,  
            flush_secs=10  
        )
        
        writer.add_text("hyperparameters", hyperparams_text)
        
        clean_log_dir = log_dir.replace('runs/', '')
        
        summary_text = f"# {model_name} on {dataset_name}\n\n"
        summary_text += f"* **Date**: {timestamp}\n"
        summary_text += f"* **Hidden Layers**: {hidden_sizes_str}\n"
        summary_text += f"* **Epochs**: {hyperparameters['NUM_EPOCHS']}\n"
        summary_text += f"* **Learning Rate**: {hyperparameters['LR']}\n"
        summary_text += f"* **Log Directory**: `{clean_log_dir}`\n"
        
        writer.add_text("model_card", summary_text)
        
        return writer, log_dir
    
    @staticmethod
    def get_logdir(writer):
        return writer.log_dir





class HyperparameterTuner:
    def __init__(self, base_hyperparameters, param_grid, language, experiment_name)

        self.base_hyperparameters = base_hyperparameters
        self.param_grid = param_grid
        self.language = language
        self.experiment_name = experiment_name
        self.results = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_model = None
        self.best_model_params = None
        self.best_test_f1 = 0.0 
        self.best_log_dir = None
        
        self._setup_dataloaders()
        
    def _setup_dataloaders(self):


        self.train_dataloader, self.test_dataloader = load_commit_history_data( 
            language=self.language,
            batch_size=self.base_hyperparameters['BATCH_SIZE'], 
            num_workers=self.base_hyperparameters['NUM_WORKERS']
        )

    def _create_param_combinations(self):

        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        param_combinations = []
        for combo in combinations:
            param_dict = self.base_hyperparameters.copy()
            for i, key in enumerate(keys):
                if key == "SCHEDULER_GAMMA":
                     param_dict["SCHEDULER"]["params"]["gamma"] = combo[i]
                else:
                    param_dict[key] = combo[i]
            param_combinations.append(param_dict)
        return param_combinations
    
    def _train_model(self, hyperparams, trial_num):

        current_batch_size = hyperparams['BATCH_SIZE']
        if self.train_dataloader and current_batch_size != self.train_dataloader.batch_size:
            self.train_dataloader, self.test_dataloader = load_commit_history_data( 
                language=self.language,
                batch_size=current_batch_size,
                num_workers=hyperparams['NUM_WORKERS']
            )
        
        hyperparams_with_language = hyperparams.copy()
        hyperparams_with_language['LANGUAGE'] = self.language
        
        model, optimizer, scheduler = create_model(hyperparams_with_language, self.device)
        
        hidden_sizes_str = "-".join([str(size) for size in hyperparams_with_language['HIDDEN_SIZES']])
        scheduler_gamma = hyperparams_with_language.get('SCHEDULER', {}).get('params', {}).get('gamma', 'N/A')
        optimizer_name = hyperparams_with_language['OPTIMIZER'].__name__
        loss_fn_name = hyperparams_with_language['LOSS_FN'].__class__.__name__
        
        optimizer_params = ""
        if optimizer_name == "Adam":
            betas = hyperparams_with_language.get('BETAS', (0.9, 0.999))
            optimizer_params = f"betas-{betas[0]}-{betas[1]}_"
        elif optimizer_name == "SGD":
            momentum = hyperparams_with_language.get('MOMENTUM', 0.0)
            optimizer_params = f"momentum-{momentum}_"
        
        description = (f"trial-{trial_num}_arch-{hidden_sizes_str}_"
                       f"batch-{hyperparams_with_language['BATCH_SIZE']}_"
                       f"lr-{hyperparams_with_language['LR']}_"
                       f"dropout-{hyperparams_with_language['DROPOUT']}_"
                       f"gamma-{scheduler_gamma}_"
                       f"opt-{optimizer_name}_"
                       f"{optimizer_params}"
                       f"loss-{loss_fn_name}_"
                       f"epochs-{hyperparams_with_language['NUM_EPOCHS']}_"
                       f"patience-{hyperparams_with_language['PATIENCE']}")

        writer, log_dir = EnhancedCustomWriter.create_Writer(
            hyperparams_with_language, 
            self.experiment_name, 
            "RelNN", 
            f"{self.language}",
            description
        )
        

        start_time = timer()
        results = train(
            model=model,
            train_dataloader=self.train_dataloader,
            test_dataloader=self.test_dataloader, 
            loss_fn=hyperparams_with_language['LOSS_FN'],
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            epochs=hyperparams_with_language['NUM_EPOCHS'],
            writer=writer,
            log_dir=log_dir,
            hyperparameters=hyperparams_with_language,
            patience=hyperparams_with_language['PATIENCE']
        )
        end_time = timer()
        train_time = end_time - start_time
        
        metrics_path = os.path.join(log_dir, 'model_artifacts', 'training_results.json')
        with open(metrics_path, 'w') as f:
             serializable_results = {}
             for key, value in results.items():
                 if isinstance(value, (list, np.ndarray)) and all(isinstance(v, (int, float, np.number)) for v in value):
                     serializable_results[key] = [float(v) for v in value]
                 elif isinstance(value, (int, float, np.number)):
                      serializable_results[key] = float(value)
                 else:
                     serializable_results[key] = value 
             json.dump(serializable_results, f, indent=4)
        writer.close()
        
        if not results.get('train_loss'):
             final_metrics = {key: float('nan') for key in [
                 'final_train_loss', 'final_train_accuracy', 'final_train_precision', 'final_train_recall', 'final_train_f1',
                 'final_test_loss', 'final_test_accuracy', 'final_test_precision', 'final_test_recall', 'final_test_f1'
             ]}
        else:
            last_epoch_idx = -1
            final_metrics = {
                'final_train_loss': results['train_loss'][last_epoch_idx],
                'final_train_accuracy': results['train_accuracy'][last_epoch_idx],
                'final_train_precision': results['train_precision'][last_epoch_idx],
                'final_train_recall': results['train_recall'][last_epoch_idx],
                'final_train_f1': results['train_f1'][last_epoch_idx],
                'final_test_loss': results['test_loss'][last_epoch_idx],
                'final_test_accuracy': results['test_accuracy'][last_epoch_idx],
                'final_test_precision': results['test_precision'][last_epoch_idx],
                'final_test_recall': results['test_recall'][last_epoch_idx],
                'final_test_f1': results['test_f1'][last_epoch_idx],
            }

        result_entry = {
            'trial': trial_num,
            'hidden_sizes': str(hyperparams_with_language['HIDDEN_SIZES']),
            'batch_size': hyperparams_with_language['BATCH_SIZE'],
            'learning_rate': hyperparams_with_language['LR'],
            'dropout': hyperparams_with_language['DROPOUT'],
            'scheduler_gamma': scheduler_gamma,
            'epochs_run': len(results.get('train_loss', [])),
            **final_metrics, 
            'training_time': train_time,
            'log_dir': log_dir
        }
        return model, result_entry, log_dir
    
    def run_tuning(self):

        param_combinations = self._create_param_combinations()
        start_trial = 1 

        for i, params in enumerate(param_combinations):
            trial_num = i + start_trial
            try:
                model, result, log_dir = self._train_model(params, trial_num)
                self.results.append(result)
                
                current_test_f1 = result.get('final_test_f1', 0.0) 
                if not np.isnan(current_test_f1) and current_test_f1 > self.best_test_f1:
                    self.best_test_f1 = current_test_f1 
                    self.best_model = model 
                    self.best_model_params = params
                    self.best_log_dir = log_dir
                    
                    best_model_path = os.path.join("best_models", f"best_model_{self.language}.pt")
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    best_params_path = os.path.join("best_models", f"best_params_{self.language}.json")
                    with open(best_params_path, 'w') as f:
                        serializable_params = {}
                        for key, value in params.items():
                             if isinstance(value, type) or callable(value): serializable_params[key] = str(value)
                             elif key == 'LOSS_FN': serializable_params[key] = value.__class__.__name__
                             elif key == 'HIDDEN_SIZES': serializable_params[key] = value
                             elif key == 'SCHEDULER':
                                 scheduler_dict = {'type': value['type'].__name__, 'params': value.get('params', {})}
                                 serializable_params[key] = scheduler_dict
                             else: serializable_params[key] = value
                        if model: serializable_params['MODEL_ARCHITECTURE'] = str(model)
                        json.dump(serializable_params, f, indent=4)


            except Exception as e:
                 self.results.append({ 
                     'trial': trial_num, 'status': 'failed', 'error': str(e), 'params': params 
                 })

            self._save_results()
        

        return {
            'best_model_state_dict_path': os.path.join("best_models", f"best_model_{self.language}.pt") if self.best_model_params else None,
            'best_hyperparameters': self.best_model_params,
            'best_test_f1_score': self.best_test_f1,
            'best_log_dir': self.best_log_dir,
            'all_results_dataframe': pd.DataFrame(self.results) 
        }
    
    def _save_results(self):

        if not self.results: return
        successful_results = [r for r in self.results if r.get('status') != 'failed']
        if not successful_results:
            results_df = pd.DataFrame(self.results) 
        else:
             results_df = pd.DataFrame(successful_results)

        results_path = f"hyperparameter_tuning_results_{self.language}.csv"
        results_df.to_csv(results_path, index=False)
        
        markdown_path = f"hyperparameter_tuning_results_{self.language}.md"
        
        sorted_results = sorted(successful_results, key=lambda x: x.get('final_test_f1', 0.0) if not np.isnan(x.get('final_test_f1', 0.0)) else 0.0, reverse=True)
        try: 
            with open(markdown_path, 'w') as f: 
                f.write(f"# Hyperparameter Tuning Results - {self.language}\n\n")
                f.write(f"Date: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                cols = ['trial', 'hidden_sizes', 'batch_size', 'learning_rate', 'dropout', 'scheduler_gamma', 
                        'epochs_run', 'final_test_f1', 'final_train_f1', 'final_test_accuracy', 
                        'final_train_accuracy', 'final_test_loss', 'final_train_loss', 'training_time']
                header = "| " + " | ".join([c.replace('_', ' ').title() for c in cols]) + " |\n"
                separator = "|-" + "-|".join(['-' * len(c.replace('_', ' ').title()) for c in cols]) + "-|\n"
                f.write(header)
                f.write(separator)
                
                for result in sorted_results: 
                    row_values = []
                    for col in cols:
                        val = result.get(col, 'N/A')
                        if isinstance(val, float): row_values.append(f"{val:.4f}")
                        else: row_values.append(str(val))
                    f.write("| " + " | ".join(row_values) + " |\n")
                
                failed_trials = [r for r in self.results if r.get('status') == 'failed']
                if failed_trials:
                    f.write("\n## Failed Trials\n\n")
                    f.write("| Trial | Error |\n")
                    f.write("|-------|-------|\n")
                    for failed in failed_trials:
                         f.write(f"| {failed.get('trial', 'N/A')} | {failed.get('error', 'Unknown')} |\n")

        except IOError as e:
             print(f"Error writing results file {markdown_path}: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for reliability prediction')
    parser.add_argument('--language', type=str, default="CPP", choices=["Java", "C", "CPP"],
                      help='Language to use (Java, C, CPP)')

    args = parser.parse_args()
    language = args.language

    base_hyperparameters = get_default_hyperparameters()
    
    param_grid = { 
        "HIDDEN_SIZES": [[256, 128, 64, 32]],
        "BATCH_SIZE": [8],
        "LEARNING_RATE": [0.001],
        "DROPOUT": [0.5],
        "SCHEDULER_GAMMA": [0.5], 
        "MOMENTUM": [0.9, 0.99],
        "BETAS": [(0.9, 0.999),],
        "LOSS_FN": [ nn.BCELoss()],
        "OPTIMIZER": [torch.optim.Adam]
    }
    os.makedirs("best_models", exist_ok=True)
    os.makedirs("runs", exist_ok=True) 

    tuner = HyperparameterTuner(
        base_hyperparameters=base_hyperparameters,
        param_grid=param_grid,
        language=language,
        experiment_name="Reliability_Prediction_Tuning" 
    )
    
    try:
        tuner.run_tuning()
    except Exception as e:
        print(f"Error during hyperparameter tuning main execution: {str(e)}")
        import traceback
        traceback.print_exc() 
        return 1

if __name__ == "__main__":
    sys.exit(main())