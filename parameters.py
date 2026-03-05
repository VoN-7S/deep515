from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List

def get_config() -> tuple[dataclass, dataclass, dataclass]:

    """
    Parses command-line arguments and returns the model, training and system configurations.

    Returns:
        tuple[ModelConfig, TrainingConfig, SystemConfig]: Configuration dataclasses
        for model architecture, training settings and system paths respectively.
    """

    params = get_params()

    model_cfg = ModelConfig(
        params.input_size,
        params.hidden_layers,
        params.activation,
        params.batch_normalization,
        params.dropout_ratio,
        params.num_classes
    )

    train_cfg = TrainingConfig(
        params.learning_rate,
        params.batch_size,
        params.epoch,
        params.regularizer,
        params.weight_decay,
        params.seed,
        params.device,
        params.log_interval,
        params.num_workers
    )

    sys_cfg = SystemConfig(
        params.loss_file,
        params.name,
        params.dataset_path,
        params.model_path,
        params.save
    )

    return model_cfg, train_cfg, sys_cfg

def get_params() -> Namespace:

    """
    Determines and parses all command-line arguments for the experiment.

    Returns:
        Namespace: Parsed argument values.
    """

    parser = ArgumentParser()
    parser.add_argument("-lf", "--loss_file", help= "Set the experiment content.", default= "random_lossfile", type=str)
    parser.add_argument("-n", "--name", help= "Set the experiment name.", default="random_experiemt", type=str)
    parser.add_argument("-is", "--input_size", help= "Set the input size.", default= 784, type=int)
    parser.add_argument("-lr", "--learning_rate", help= "Set the learning rate.", default=1e-3 ,type=float)
    parser.add_argument("-dr", "--dropout_ratio", help= "Set the dropout_ratio.", default= 0, type=float)
    parser.add_argument("-hl", "--hidden_layers", nargs= '+', help = "Set the hidden layer size.", default= [196, 49], type=int)
    parser.add_argument("-bs", "--batch_size", help= "Set the batch size.", default=64, type=int)
    parser.add_argument("-dsp", "--dataset_path", help= "Set the dataset path.", default= "./data", type=str)
    parser.add_argument("-mp", "--model_path", help= "Set the model path.", default="best_model.pth", type=str)
    parser.add_argument("-a", "--activation", help= "Set the activation function among.", choices=["r", "lr", "s", "t"], default="r", type=str)
    parser.add_argument("-bn", "--batch_normalization", help= "Set batch normalization.", action= "store_true")
    parser.add_argument("-e", "--epoch", help= "Set the # of epochs.", default=25, type=int)
    parser.add_argument("-r", "--regularizer", help= "Set the regularizer.", choices=[1, 2], default=2, type=int)
    parser.add_argument("-wd", "--weight_decay", help= "Set the weight decay.", default=0.0, type=float)
    parser.add_argument("-d", "--device", help= "Set the device.", default="cpu", type=str)
    parser.add_argument("-s", "--seed", help= "Set the seed.", default=7, type=int)
    parser.add_argument("-nw", "--num_workers", help= "Set the number of workers.", default=0, type=int)
    parser.add_argument("-li", "--log_interval", help= "Set the batch interval for log print.", default=100, type=int)
    parser.add_argument("-nc", "--num_classes", help= "Set the final number of classes.", default=10, type=int)
    parser.add_argument("-sv", "--save", help= "Set whether the results will be saved.", action="store_true")

    return parser.parse_args()


@dataclass
class ModelConfig:
    """Determines the MLP architecture."""
    input_size: int           # number of input features (784 for MNIST)
    hidden_layers: List[int]  # neuron count per hidden layer e.g. [196, 49]
    activation: str           # activation function (r=ReLU, lr=LeakyReLU, s=Sigmoid, t=Tanh)
    batch_normalization: bool # whether to apply batch normalization after each layer
    dropout_ratio: float      # dropout probability (0 disables dropout)
    num_classes: int          # number of output classes (10 for MNIST)

@dataclass
class TrainingConfig:
    """Determines the training loop settings."""
    learning_rate: float  # step size for optimizer
    batch_size: int       # number of samples per mini-batch
    epoch: int            # total number of training epochs
    regularizer: int      # regularization type (1=L1, 2=L2)
    weight_decay: float   # regularization strength (0 disables regularization)
    seed: int             # random seed for reproducibility
    device: str           # device to train on (e.g. 'cpu', 'cuda')
    log_interval: int     # print logs every N batches
    num_workers: int      # number of DataLoader worker processes


@dataclass
class SystemConfig:
    """Determines system-level paths and saving behavior."""
    loss_file: str    # filename to save loss curves
    name: str         # experiment name for logging
    dataset_path: str # directory to download/load MNIST data
    model_path: str   # path to save or load model checkpoint
    save: bool        # whether to save results to disk


