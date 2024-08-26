import logging

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models, transforms
import gc
from torchmetrics import MeanSquaredError, Accuracy

from models.blur_pool.BlurPoolConv2d import apply_blurpool

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelLightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task_type = self.hparams.task_type

        self.model = self.get_model()
        logger.info(f"Model architecture: {self.hparams.arch}")
        logger.debug(f"Model structure: {self.model}")

        # Create sample input
        self.example_input_array = self.create_sample_input()
        logger.debug(
            f"Created sample input with shape: {self.example_input_array.shape}"
        )

        self.setup_task_specific_components()

    def create_sample_input(self):
        # Adjust input size based on your model's requirements
        if self.task_type in ["classification", "regression", "combined"]:
            return torch.randn(1, 3, 224, 224)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def setup_task_specific_components(self):
        if self.task_type in ["classification", "combined"]:
            self.classification_criterion = nn.CrossEntropyLoss()
            self.accuracy = Accuracy(
                task="multiclass",
                num_classes=self.hparams.num_classes,
                top_k=1,
            )

        if self.task_type in ["regression", "combined"]:
            self.regression_criterion = nn.MSELoss()
            self.mse = MeanSquaredError()

        if self.task_type not in ["classification", "regression", "combined"]:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        print(f"Set up components for {self.task_type} task.")

    def forward(self, x):
        if self.task_type == "classification":
            return self.model(x)
        elif self.task_type == "regression":
            return self.model(x)
        elif self.task_type == "combined":
            if self.hparams.experiment == "one":
                features = self.model(x)
                return self.classification_head(features), self.regression_head(
                    features
                )
            elif self.hparams.experiment == "two":
                features = self.feature_extractor(x)
                features = features.view(features.size(0), -1)
                return self.classification_head(features), self.regression_head(
                    features
                )

    def unpack_batch(self, batch):
        if isinstance(batch, dict):
            if "classification" in batch and "regression" in batch:
                return batch["classification"], batch["regression"]
            elif "classification" in batch:
                return batch["classification"], None
            elif "regression" in batch:
                return None, batch["regression"]
        return batch, None

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        # print("-"* 100)
        # print(f"{dataloader_idx = }")
        # print(f"{batch_idx = }")
        # print(f"{len(batch) = }")
        # print(f"{type(batch) = }")
        # print(f"{batch.keys() = }")

        return self.shared_step(batch, "train", dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self.shared_step(batch, "val", dataloader_idx)

    def shared_step(self, batch, stage, dataloader_idx=None):
        if self.task_type == "combined":
            return self.combined_step(batch, stage, dataloader_idx)
        elif self.task_type == "classification":
            return self.classification_step(batch, stage)
        elif self.task_type == "regression":
            return self.regression_step(batch, stage)

    def combined_step(self, batch, stage, dataloader_idx):
        class_batch, reg_batch = self.unpack_batch(batch)

        losses = []
        if class_batch is not None:
            class_loss = self.classification_step(
                class_batch, stage, task_prefix="combined_class"
            )
            losses.append(class_loss)

        if reg_batch is not None:
            reg_loss = self.regression_step(
                reg_batch, stage, task_prefix="combined_reg"
            )
            losses.append(reg_loss)

        if not losses:
            raise ValueError(f"Invalid batch structure: {batch}")

        total_loss = sum(losses)
        self.log(
            f"{stage}/combined_total_loss",
            total_loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def handle_combined_output(self, outputs, task):
        if isinstance(outputs, tuple):
            if task == "classification":
                return outputs[0]  # Assume classification output is first
            elif task == "regression":
                return outputs[1]  # Assume regression output is second
            else:
                raise ValueError(f"Invalid task: {task}")
        return outputs

    def classification_step(self, batch, stage, task_prefix="classification"):
        inputs, targets = batch
        outputs = self(inputs)

        classification_outputs = self.handle_combined_output(outputs, "classification")
        loss, acc = self.classification_loss(classification_outputs, targets)

        self.log_metrics(
            {
                f"{stage}/{task_prefix}_loss": loss,
                f"{stage}/{task_prefix}_accuracy": acc,
            },
            stage,
        )

        return loss

    def regression_step(self, batch, stage, task_prefix="regression"):
        inputs, targets = batch
        outputs = self(inputs)

        regression_outputs = self.handle_combined_output(outputs, "regression")
        loss, mse = self.regression_loss(regression_outputs, targets)

        self.log_metrics(
            {f"{stage}/{task_prefix}_loss": loss, f"{stage}/{task_prefix}_mse": mse},
            stage,
        )

        return loss

    def classification_loss(self, outputs, targets):
        loss = self.classification_criterion(outputs, targets)
        acc = self.accuracy(outputs, targets)
        return loss, acc

    def regression_loss(self, outputs, targets):
        loss = self.regression_criterion(outputs, targets)
        mse = self.mse(outputs, targets)
        return loss, mse

    def log_metrics(self, metrics, stage):
        on_step = True if stage == "train" else False
        on_epoch = True
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if self.task_type == "combined":
                # Use the sum of classification and regression losses for combined tasks
                scheduler_config["monitor"] = "val/combined_total_loss"
            elif self.task_type == "classification":
                scheduler_config["monitor"] = "val/classification_loss"
            elif self.task_type == "regression":
                scheduler_config["monitor"] = "val/regression_loss"
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def get_optimizer(self):
        # Convert parameters to appropriate types
        lr = float(self.hparams.lr)
        weight_decay = float(self.hparams.weight_decay)

        if self.hparams.optimizer.lower() == "adam":
            return optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif self.hparams.optimizer.lower() == "sgd":
            momentum = float(self.hparams.momentum)
            return optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=self.hparams.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

    def get_scheduler(self, optimizer):
        if self.hparams.scheduler == "step":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.step_size,
                gamma=self.hparams.lr_gamma,
            )
        elif self.hparams.scheduler == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            )
        elif self.hparams.scheduler == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

    def get_model(self):
        logger.info(f"Creating model with architecture: {self.hparams.arch}")
        logger.info(f"Pretrained: {self.hparams.pretrained}")

        model = self._get_model_by_arch(self.hparams.arch, self.hparams.pretrained)
        logger.debug(f"Base model structure:\n{model}")

        # Get the number of features in the last layer
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
            else:
                in_features = model.classifier.in_features
        elif hasattr(model, "heads"):  # For ViT models
            in_features = model.heads.head.in_features
        else:
            raise AttributeError(
                f"Unable to determine in_features for model {self.hparams.arch}"
            )

        if self.task_type == "regression":
            logger.info("Modifying model for regression task")
            new_head = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
        elif self.task_type == "classification":
            logger.info("Modifying model for classification task")
            new_head = nn.Linear(in_features, self.hparams.num_classes)
        elif self.task_type == "combined":
            logger.info("Modifying model for combined task")
            if self.hparams.experiment == "one":
                logger.info("Using experiment one configuration")
                new_head = nn.Identity()
                self.classification_head = nn.Linear(
                    in_features, self.hparams.num_classes
                )
                self.regression_head = nn.Sequential(
                    nn.Linear(in_features, 1), nn.Sigmoid()
                )
                logger.debug(f"Classification head: {self.classification_head}")
                logger.debug(f"Regression head: {self.regression_head}")
            elif self.hparams.experiment == "two":
                logger.info("Using experiment two configuration")
                self.feature_extractor = self.create_feature_extractor(
                    model, self.hparams.feature_layer
                )
                logger.debug(
                    f"Created feature extractor up to layer: {self.hparams.feature_layer}"
                )
                feature_size = self._get_feature_size(model)
                logger.info(f"Feature size: {feature_size}")
                self.classification_head = nn.Linear(
                    feature_size, self.hparams.num_classes
                )
                self.regression_head = nn.Sequential(
                    nn.Linear(feature_size, 1), nn.Sigmoid()
                )
                logger.debug(f"Classification head: {self.classification_head}")
                logger.debug(f"Regression head: {self.regression_head}")
                return model  # Early return for experiment two

        # Replace the last layer
        if hasattr(model, "fc"):
            model.fc = new_head
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                model.classifier[-1] = new_head
            else:
                model.classifier = new_head
        elif hasattr(model, "heads"):  # For ViT models
            model.heads.head = new_head
        else:
            raise AttributeError(
                f"Unable to modify last layer for model {self.hparams.arch}"
            )

        logger.debug(f"Modified last layer: {new_head}")

        if self.hparams.use_blurpool:
            logger.info("Applying BlurPool to the model")
            apply_blurpool(model)
            logger.debug("BlurPool applied")

        logger.info("Model creation completed")
        logger.debug(f"Final model structure:\n{model}")
        return model

    def create_feature_extractor(self, model, layer_name):
        logger.debug(f"Creating feature extractor up to layer: {layer_name}")
        return nn.Sequential(
            *list(model.children())[: self._get_layer_index(model, layer_name)]
        )

    def _get_layer_index(self, model, layer_name):
        for i, (name, _) in enumerate(model.named_children()):
            if name == layer_name:
                logger.debug(f"Found layer {layer_name} at index {i}")
                return i
        logger.error(f"Layer {layer_name} not found in the model")
        raise ValueError(f"Layer {layer_name} not found in the model")

    def _get_feature_size(self, model):
        logger.debug("Calculating feature size")
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Adjust input size if necessary
            features = self.feature_extractor(dummy_input)
            feature_size = features.numel()
            logger.debug(f"Calculated feature size: {feature_size}")
            return feature_size

    def _get_model_by_arch(self, arch, pretrained):
        logger.debug(f"Getting model for architecture: {arch}")
        weights = "DEFAULT" if pretrained else None
        logger.debug(f"Using weights: {weights}")

        if arch == "resnet18":
            return models.resnet18(weights=weights)
        elif arch == "resnet50":
            return models.resnet50(weights=weights)
        elif arch == "resnet101":
            return models.resnet101(weights=weights)
        elif arch == "alexnet":
            return models.alexnet(weights=weights)
        elif arch == "vgg16":
            return models.vgg16(weights=weights)
        elif arch == "vgg19":
            return models.vgg19(weights=weights)
        elif arch == "efficientnet_b0":
            return models.efficientnet_b0(weights=weights)
        elif arch == "vit_b_16":
            return models.vit_b_16(weights=weights)
        elif arch == "vit_b_32":
            return models.vit_b_32(weights=weights)
        elif arch == "inception_v3":
            return models.inception_v3(weights=weights)
        else:
            logger.error(f"Unknown architecture: {arch}")
            raise ValueError(f"Unknown architecture: {arch}")

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("grad_norm", grad_norm)

    def on_train_epoch_end(self):
        gc.collect()
