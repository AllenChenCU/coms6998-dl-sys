import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import json

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = ["GoogLeNet", "GoogLeNetOutputs", "_GoogLeNetOutputs", "GoogLeNet_Weights", "googlenet"]


GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of GoogleNet will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.conv1_module = conv_block(3, 96, kernel_size=3, stride=1)
        self.inception2_1_module = InceptionModule(96, 32, 32)
        self.inception2_2_modeul = InceptionModule(64, 32, 48)
        self.downsample2_module = DownsampleModule(80, 80, 1)
        self.inception3_1_module = InceptionModule(160, 112, 48)
        self.inception3_2_module = InceptionModule(160, 96, 64)
        self.inception3_3_module = InceptionModule(160, 80, 80)
        self.inception3_4_module = InceptionModule(160, 48, 96)
        self.downsample3_module = DownsampleModule(144, 96, 0)
        self.inception4_1 = InceptionModule(240, 176, 160)
        self.inception4_2 = InceptionModule(336, 176, 160)
        #self.mean_pooling4 = nn.AvgPool2d(7)
        self.fc4 = nn.Linear(336, num_classes)
        
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x: Tensor) -> Tensor: #Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

        # N x 1 x 28 X 28
        x = self.conv0(x)
        # N x 3 x 28 X 28
        x = self.conv1_module(x)
        x = self.inception2_1_module(x)
        x = self.inception2_2_modeul(x)
        x = self.downsample2_module(x)
        x = self.inception3_1_module(x)
        x = self.inception3_2_module(x)
        x = self.inception3_3_module(x)
        x = self.inception3_4_module(x)
        x = self.downsample3_module(x)
        x = self.inception4_1(x)
        x = self.inception4_2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class InceptionModule(nn.Module):
    def __init__(
        self, 
        in_channel, 
        ch1,
        ch3
    ):
        super().__init__()
        self.conv_module1 = BasicConv2d(in_channel, ch1, kernel_size=1, stride=1)
        self.conv_module2 = BasicConv2d(in_channel, ch3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv_module1(x)
        x2 = self.conv_module2(x)
        return torch.cat([x1, x2], 1)


class DownsampleModule(nn.Module):
    def __init__(
        self, 
        in_channel, 
        ch3, 
        padding
    ):
        super().__init__()
        self.conv_module = BasicConv2d(in_channel, ch3, kernel_size=3, stride=2, padding=padding)
        self.max_pool = nn.MaxPool2d(3, stride=2, ceil_mode=True, padding=0)
    
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv_module(x)
        x2 = self.max_pool(x)
        return torch.cat([x1, x2], 1)


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class GoogLeNet_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/googlenet-1378be20.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 6624904,
            "min_size": (15, 15),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#googlenet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.778,
                    "acc@5": 89.530,
                }
            },
            "_ops": 1.498,
            "_file_size": 49.731,
            "_docs": """These weights are ported from the original paper.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


#@register_model()
@handle_legacy_interface(weights=("pretrained", GoogLeNet_Weights.IMAGENET1K_V1))
def googlenet(*, weights: Optional[GoogLeNet_Weights] = None, progress: bool = True, **kwargs: Any) -> GoogLeNet:
    """GoogLeNet (Inception v1) model architecture from
    `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

    Args:
        weights (:class:`~torchvision.models.GoogLeNet_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.GoogLeNet_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.GoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
    """
    weights = GoogLeNet_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GoogLeNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        else:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )

    return model


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, scheduler=None, device=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
        #self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.criterion = nn.NLLLoss() if criterion is None else criterion
        self.scheduler = scheduler
        self.device = "cpu" if device is None else device

        self.model = self.model.to(self.device)

    def train(self, epochs, train_dataloader, val_dataloader=None):

        results = {
            "train_loss": [],
            "train_acc": [],
            "train_batch_losses": [], 
            "train_batch_accs": [], 
        }
        if val_dataloader:
            results["val_loss"] = []
            results["val_acc"] = []
            results["val_batch_losses"] = []
            results["val_batch_accs"] = []

        #batch_size = 32
        # Iterate over epochs
        for epoch in tqdm(range(epochs)):
            # Training the epoch
            #train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
            train_loss, train_acc, train_batch_losses, train_batch_accs = self._train_epoch(
                dataloader=train_dataloader,
            )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["train_batch_losses"].extend(train_batch_losses)
            results["train_batch_accs"].extend(train_batch_accs)

            # eval the epoch
            if val_dataloader:
                #val_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
                val_loss, val_acc, val_batch_losses, val_batch_accs = self._eval_epoch(
                    dataloader=val_dataloader,
                )
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)
                results["val_batch_losses"].extend(val_batch_losses)
                results["val_batch_accs"].extend(val_batch_accs)
            #batch_size *= 2
        return results

    def _train_epoch(self, dataloader):
        # Training mode
        self.model.train()

        # Iterate through batches
        total_samples = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_losses = []
        batch_accs = []
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            # train steps
            probs = self.model(x)
            loss = self.criterion(probs, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                scheduler.step()

            # compute metrics
            num_in_batch = len(x)
            total_samples += num_in_batch
            epoch_loss += loss.detach().item() * num_in_batch
            batch_losses.append(loss.detach().item())
            #probs = F.softmax(logits, dim=1)
            preds = torch.max(probs, dim=1).indices
            acc = torch.sum(torch.eq(preds, y)).detach().item()
            batch_accs.append(acc/num_in_batch)
            epoch_acc += acc

        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_acc = epoch_acc / total_samples
        return avg_epoch_loss, avg_epoch_acc, batch_losses, batch_accs

    def _eval_epoch(self, dataloader):

        # Eval mode
        self.model.eval()

        # Iterate through batches
        total_samples = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_losses = []
        batch_accs = []
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                probs = self.model(x)
                loss = self.criterion(probs, y)

                # compute metrics
                num_in_batch = len(x)
                total_samples += num_in_batch
                epoch_loss += loss.detach().item() * num_in_batch
                batch_losses.append(loss.detach().item())
                #probs = F.softmax(logits, dim=1)
                preds = torch.max(probs, dim=1).indices
                acc = torch.sum(torch.eq(preds, y)).detach().item()
                batch_accs.append(acc/num_in_batch)
                epoch_acc += acc

        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_acc = epoch_acc / total_samples
        return avg_epoch_loss, avg_epoch_acc, batch_losses, batch_accs

## Small model for testing preliminary code
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    # BATCH_SIZE = 64
    BATCH_SIZE = 32
    # EPOCHS = 5
    EPOCHS = 9

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), 
        ]
    )

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

    model = GoogLeNet(
        num_classes=10,
        aux_logits=True,
        transform_input=False,
        init_weights=None,
        blocks=None,
        dropout=0.2,
        dropout_aux=0.7,
    )
    #model = GarmentClassifier()
    criterion = torch.nn.CrossEntropyLoss()

    # # prob3.1 
    # lrs = []
    # lr = 0.0025
    # for _ in range(10):
    #     lrs.append(lr)
    #     lr += 0.0025
    # prob3_1_results = {}
    # for i, lr in enumerate(tqdm(lrs)):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    #     trainer = Trainer(model, optimizer=optimizer, criterion=criterion) #, device="cuda")

    #     results = trainer.train(EPOCHS, training_loader, validation_loader)
    #     results['lr'] = lr
    #     prob3_1_results[str(i)] = results
    
    # with open('prob3_1_results.json', 'w') as fp:
    #     json.dump(prob3_1_results, fp)
    

    # prob3.2
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0025, max_lr=0.01)
    # trainer = Trainer(model, optimizer=optimizer, criterion=criterion, scheduler=scheduler)
    # prob3_2_results = trainer.train(EPOCHS, training_loader, validation_loader)
    # with open('prob3_2_results.json', 'w') as fp:
    #     json.dump(prob3_2_results, fp)


    # prob3.3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer=optimizer, criterion=criterion) #, device="cuda")
    prob3_3_results = trainer.train(EPOCHS, training_loader, validation_loader)

    with open('prob3_3_results.json', 'w') as fp:
        json.dump(prob3_3_results, fp)
