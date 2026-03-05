import matplotlib.pyplot as plt
import pandas as pd
import os

from parameters import TrainingConfig, ModelConfig, SystemConfig


def save_training_loss(training_config: TrainingConfig, model_config: ModelConfig, sys_config: SystemConfig, tloss_list: list, vloss_list: list) -> None:
    
     """
     Tracks the training and validation loss.

     Args:
          training_config(TrainingConfig): Training configurations.  Contains number of epochs.
          model_config(ModelConfig): Model configurations.
          sys_config(SystemConfig): System configurations. Contains file path for saving training and validation loss graph.
          tloss_list(list): Training loss over epochs.
          vloss_list(list): Validation loss over epochs.
     """

     important_parameters = ["hidden_layers", "batch_normalization", "activation", "learning_rate", "dropout_ratio", "batch_size", "regularizer", "weight_decay", "seed"] # Determine the model parameters to be saved.
     # Plot training and validation loss over epochs.
     plt.plot(range(1, training_config.epoch + 1), tloss_list, color="r", label= "Training Loss")
     plt.plot(range(1, training_config.epoch + 1), vloss_list, color='0', label= "Validation Loss")
     # Extract the important parameters.
     arg_dict = {**vars(training_config), **vars(model_config), **vars(sys_config)}
     arg_str = "\n".join([f"{k}: {arg_dict[k]}" for k in important_parameters])
     # Put the important parameters on the plot.
     plt.text(0.95, 0.95, arg_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
     # Create the save file.
     os.makedirs("./results/" + sys_config.loss_file, exist_ok=True)
     # Save the parameters in an excel format.
     pd.DataFrame([arg_dict]).to_excel("./results/" + arg_dict["loss_file"] + "/" + arg_dict["name"] + ".xlsx")
     
     plt.xlabel("Epoch #")
     plt.ylabel("Loss")
     plt.ylim(bottom= 0)
     plt.legend(loc="lower left")
     plt.grid()
     plt.savefig("./results/" + arg_dict["loss_file"] + "/" + arg_dict["name"] + ".png")