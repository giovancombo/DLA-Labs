import torch
import yaml
import wandb

import pipeline as pipe
import utils

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login()
    print("Initializing Weights & Biases run...")

    # Loading the configuration file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Initializing a wandb run for logging losses, accuracies and gradients
    with wandb.init(project = config['project_name'], config = config):
        config = wandb.config

        # 1. Loading the data
        train_loader, val_loader, test_loader = pipe.load(config)

        # 2. Building the model
        model, criterion, optimizer = pipe.build_model(device, config)

        # 3. Training the model
        pipe.train(model, train_loader, val_loader, criterion, optimizer, device, config)

        # 4. Evaluate the model on the test set
        test_loss, test_accuracy = pipe.test(model, test_loader, device, config)

        print(f"Testing completed! | Test Loss: {test_loss:.4f}; Test Accuracy = {test_accuracy:.2f}%")
        wandb.log({"Test Loss": test_loss,
                "Test Accuracy": test_accuracy})
        
        # 5. Saving the model, assigning it a name based on the hyperparameters used
        if config['save_model']:
            utils.save_model(config, model)


if __name__ == "__main__":
    
    main()