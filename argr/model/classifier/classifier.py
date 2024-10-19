from transformers import ViTModel
from torch import nn

class ViT4Clf(nn.Module):
    """
    Vision Transformer (ViT) based model for image classification.

    Args:
        num_labels (int): Number of labels for classification.
        pretrained_model_name (str): Name of the pretrained ViT model to be used.
    """
    def __init__(self, num_labels: int = 13, pretrained_model_name: str = "google/vit-base-patch16-224"):
        super(ViT4Clf, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        """
        Forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output logits for classification.
        """
        outputs = self.vit(pixel_values=pixel_values)
        hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.classifier(hidden_state)
        return logits