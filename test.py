import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import TrainingConfig, SystemConfig, ModelConfig
from models.MNISTModels import MLPBase



@torch.no_grad() # Disable gradient calculation for efficiency.
def test(model: MLPBase, training_config: TrainingConfig, system_config: SystemConfig, model_config: ModelConfig) -> None:

    """
    Evaluates the trained model on the MNIST test set and reports overall
    and per-class accuracy.

    Args:
        model (MLPBase): The MLP model to evaluate.
        training_config (TrainingConfig): Contains batch size, number of workers and device.
        system_config (SystemConfig): Contains model checkpoint and dataset paths.
        model_config (ModelConfig): Contains number of output classes.
    """

    # Load the saved model checkpoint
    model.load_state_dict(torch.load(system_config.model_path, map_location=training_config.device))
    model.eval()
    # Normalize the dataset.
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)
    ])

    test_ds = datasets.MNIST(
        root= system_config.dataset_path,
        train= False,
        transform= test_tf,
        download=True
    )

    test_loader = DataLoader(test_ds, training_config.batch_size, True, num_workers=training_config.num_workers)


    acc = 0 # Total accuracy.
    class_acc = [0] * model_config.num_classes # Correct predictions per class.
    class_total = [0] * model_config.num_classes # Total number per class.

    for batch_idx, (batch, labels) in enumerate(test_loader):

        pred = torch.argmax(model(batch), 1)
        acc += torch.sum(torch.eq(pred, labels)) # Accumulate total accuracy.

        #Accumulate per class accuracy.
        for prediction, label in zip(pred, labels):
            class_acc[label] += prediction == label
            class_total[label] += 1
            
    # Print the validation results.
    print("\n===== TEST RESULTS =====\n")
    print(f"Overall Accuracy: {acc} / {len(test_ds)} = {acc / len(test_ds): .4f}")
    for i in range(len(class_total)):
        print(f"Digit #{i} -> Accuracy: {class_acc[i] / class_total[i]: .4f}")





