import yaml
from pathlib import Path


def generate_config(model, params):
    base_config = {
        "task_type": "classification",
        "max_epochs": params["epochs"],
        "arch": model,
        "optimizer": params.get("optimizer", "sgd"),
        "num_classes": 1000,
        "image_size": 224,
        "batch_size": params["batch_size"],
        "num_workers": 4,
        "data_dir": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/lamem/lamem_images/lamem",
        "temp_extract": True,
        "change_labels": False,
        "pin_memories": [False, False, False],
        "pretrained": False,
        "random_training": True,
        "use_blurpool": True,
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "momentum": params["momentum"],
        "nesterov": True,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "experiment": "one",
        "scheduler": params["scheduler"],
        "step_size": params["step_size"],
        "lr_gamma": params["lr_gamma"],
        "limit_train_batches": 20,
        "limit_val_batches": 10,
        "log_every_n_steps": 1,
        "accelerator": "auto",
        "strategy": "auto",
        "devices": "auto",
        "fast_dev_run": False,
        "sync_batchnorm": True,
        "num_nodes": 1,
        "log_dir": "experiments",
    }

    return base_config


def save_config(config, filename):
    with open(filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# Define model groups and their parameters
model_groups = {
    "group1": {
        "models": ["alexnet", "resnet18", "resnet50", "resnet101", "vgg16", "vgg19"],
        "params": {
            "batch_size": 256,
            "epochs": 90,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "step_size": 30,
            "lr_gamma": 0.1,
            "scheduler": "step",
            "optimizer": "adam",
        },
    },
    "group2": {
        "models": ["efficientnet_b0"],
        "params": {
            "batch_size": 128,
            "epochs": 600,
            "lr": 0.5,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "step_size": 30,
            "lr_gamma": 0.1,
            "scheduler": "cosine",
            "optimizer": "adam",
        },
    },
    "group3": {
        "models": ["vit_b_16", "vit_b_32"],
        "params": {
            "batch_size": 512,
            "epochs": 300,
            "lr": 0.003,
            "momentum": 0.9,
            "weight_decay": 0.3,
            "step_size": 30,
            "lr_gamma": 0.1,
            "scheduler": "cosine",
            "optimizer": "adam",
            "image_size": 224,
        },
    },
}

# Create output directory
output_dir = Path("config_files")
output_dir.mkdir(exist_ok=True)

# Generate and save config files for each model
for group, group_data in model_groups.items():
    for model in group_data["models"]:
        config = generate_config(model, group_data["params"])
        filename = output_dir / f"{config['task_type']}_{model}_config.yaml"
        save_config(config, filename)
        print(f"Generated config file for {model}: {filename}")

print("All configuration files have been generated.")
