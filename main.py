from copy import deepcopy
import torch
import torch.nn as nn
import ssl
from torchsummary import summary

from models.MNISTModels import MLPBase
from train import *
from parameters import get_config
from test import test
from auxillary import *

ssl._create_default_https_context = ssl._create_unverified_context # Fixes macOS dataset download problem.

model_cfg, training_cfg, sys_cfg = get_config() # Get the necessary configurations to access all pipeline hyperparameters.

set_seed(training_cfg.seed)
device = "cuda" if (training_cfg.device == "gpu" and torch.cuda.is_available()) else "cpu" # Set cpu as the device in case cuda not available.

model = MLPBase(model_cfg).to(device) # Attach the model to the device.

summary(model, (model_cfg.input_size, )) # Summarize the model
print("\n")

tloader, vloader = get_loaders(training_cfg, sys_cfg) 

best_weights = None
best_acc = 0
training_loss_tracker = []
training_acc_tracker = []
val_loss_tracker = []
val_acc_tracker = []

for epoch in range(training_cfg.epoch):
    print(f"\nEPOCH #{epoch + 1}\n")

    criterion = nn.CrossEntropyLoss(reduction='sum') # Directly calculate the sum of the losses.
    opt = torch.optim.SGD(model.parameters(), lr=training_cfg.learning_rate)
    training_loss, training_acc = train_one_epoch(training_cfg, model, criterion, opt, device, tloader)
    val_loss, val_acc = validate(model, criterion, device, vloader)

    training_loss_tracker.append(training_loss)
    training_acc_tracker.append(training_acc)
    val_loss_tracker.append(val_loss)
    val_acc_tracker.append(val_acc)

    print(f"\nAverage Training Loss: {training_loss: .4f} |==| Average Validation Loss: {val_loss: .4f}")
    print(f"Average Training Accuracy: {training_acc: .4f} |==| Average Validation Accuracy: {val_acc: .4f}")
    # Save the model weights with the least validation loss (best validation accuracy).
    if val_acc > best_acc:
        best_weights = deepcopy(model.state_dict())
        print("Best Model Saved!")
        best_acc = val_acc


torch.save(best_weights, sys_cfg.model_path) # Save the weights to the model path.
test(model, training_cfg, sys_cfg, model_cfg)  # Test the best model.
if sys_cfg.save: # Save training and validation loss curves if instructed.
    save_training_loss(training_cfg, model_cfg, sys_cfg, training_loss_tracker, val_loss_tracker)


