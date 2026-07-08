import torch
import torchvision.models as models

class CNN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(
                self, 
                in_channels: int, 
                out_channels: int, 
                n_layers: int, 
                stride: int = 2, 
                kernel_size: int = 3
            ):
            """
            A block of layers in the convolutional network

            Args:
                in_channels: int, number of input channels
                out_channels: int, number of output channels
                n_layers: int, number of convolutional layers in block
                stride: int, stride for first convolutional layer
                kernel_size: int, kernel size for all convolutional layers
            """
            super().__init__()

            assert n_layers >= 1

            padding = (kernel_size - 1) // 2

            self.conv_layers = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            for i in range(n_layers):
                layer_in = in_channels if i == 0 else out_channels
                layer_stride = stride if i == 0 else 1

                self.conv_layers.append(
                    torch.nn.Conv2d(layer_in, out_channels, kernel_size, layer_stride, padding)
                )
                self.batch_norms.append(
                    torch.nn.BatchNorm2d(out_channels)
                )

            
            self.relu = torch.nn.ReLU()
            self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        def forward(self, x):
            identity = self.skip(x)
            for conv, bn in zip(self.conv_layers, self.batch_norms):
                x = self.relu(bn(conv(x)))
            x = x + identity # residual connection
            return self.relu(x)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        n_blocks: int = 3,
        n_layers_per_block: int = 3,
        channels_l0: int = 16,
        stem_kernel_size: int = 11,
        stem_stride: int = 2,
        block_stride: int = 2,
        channel_growth_factor: int = 2,
        dropout_p: float = 0.3
    ):
        """
        A convolutional network built from scratch for image classification.

        Args:
            in_channels: int, classes of pixels (e.g. RGB)
            num_classes: int, classes of skin lesion types we are trying to classify
            n_blocks: int, number of blocks in the network
            n_layers_per_block: int, number of convolutional layers per block
            channels_l0: int, number of channels in the first block, the number of channels grows by channel_growth_factor in each subsequent block
            stem_kernel_size: int, kernel_size of first convolutional layer
            stem_stride: int, stride of first convolutional layer
            block_stride: int, stride of first layer in each block
            channel_growth_factor: int, factor at which channels grow between blocks
            dropout_p: float, dropout percentage before last linear layer
        """
        super().__init__()

        stem_padding = (stem_kernel_size - 1) // 2

        cnn_layers = [
            torch.nn.Conv2d(in_channels, channels_l0, kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding),
            torch.nn.ReLU(),
        ]
        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * channel_growth_factor
            cnn_layers.append(self.Block(in_channels=c1, out_channels=c2, n_layers=n_layers_per_block, stride=block_stride))
            c1 = c2

        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1)) # Pooling layer that averages every value across the entire HxW grid per channel, Reshaping from (batch, channels, H, W) to (batch, channels, 1, 1)
        cnn_layers.append(torch.nn.Flatten()) # Reshapes from (batch, channels, 1, 1) to (batch, channels)
        cnn_layers.append(torch.nn.Dropout(p=dropout_p)) # Dropout layer randomly zeroes out (dropout_p * 100)% of values passing through to help with overfitting
        cnn_layers.append(torch.nn.Linear(c1, num_classes)) # Classifier layer that  maps to a num_classes sized 1D tensor of logits that we generate a prediction from
        self.network = torch.nn.Sequential(*cnn_layers)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 6} with shape (b,)
        """
        return self(x).argmax(dim=1)
    
class ResNetFrozen(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 7
    ):
        """
        A ResNet-18 backbone pretrained on ImageNet, with all backbone weights frozen and a new trainable linear classifier head.

        Args:
        num_classes: int, number of output. In our case classes of skin lesion types we are trying to classify
        """
        super().__init__()

        # Load ResNet-18 pre-trained model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 6} with shape (b,)
        """
        return self(x).argmax(dim=1)