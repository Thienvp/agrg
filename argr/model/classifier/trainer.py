from time import time
import torch
import wandb

class ViTClassifierTrainer:
    """
    Trainer class for Vision Transformer (ViT) model for chest X-ray classification.

    Args:
        model (nn.Module): The model to be trained.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to train the model on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
        accumulation_steps (int): Number of gradient accumulation steps.
        logging (bool): Flag to enable logging (e.g., with wandb).
    """
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, device, num_epochs, accumulation_steps, logging=False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.logging = logging

    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0

        for i, batch in enumerate(self.train_dataloader):
            pixel_values, labels = self._move_to_device(batch)

            if pixel_values is None or labels is None:
                continue

            outputs = self.model(pixel_values)
            loss = self.criterion(outputs, labels) / self.accumulation_steps
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

            if self.logging:
                wandb.log({"batch_loss": loss.item() * self.accumulation_steps})

        return total_loss / len(self.train_dataloader)

    def train(self):
        """
        Train the model for multiple epochs.
        """
        for epoch in range(self.num_epochs):
            start_time = time()
            
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            self._save_checkpoint(epoch)
            
            epoch_duration = time() - start_time
            self._log_epoch(epoch, train_loss, val_loss, epoch_duration)
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                pixel_values, labels = self._move_to_device(batch)

                if pixel_values is None or labels is None:
                    continue

                outputs = self.model(pixel_values)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def _move_to_device(self, batch):
        """
        Move a batch of data to the specified device.

        Args:
            batch (tuple): Batch of data.

        Returns:
            tuple: Batch of data moved to the specified device.
        """
        pixel_values, labels = batch
        return pixel_values.to(self.device), labels.to(self.device)

    def _save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, f"vit4abclf_epoch_{epoch}.pth")

    def _log_epoch(self, epoch, train_loss, val_loss, duration):
        """
        Log information for each epoch.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            val_loss (float): Validation loss for the epoch.
            duration (float): Duration of the epoch in seconds.
        """
        print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {duration:.2f}s")
        if self.logging:
            wandb.log({
                "epoch_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]['lr'],
                "epoch_duration": duration
            })
