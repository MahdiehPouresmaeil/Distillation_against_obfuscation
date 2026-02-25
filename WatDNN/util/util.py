import os
import itertools as it
import random
from copy import deepcopy
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

from networks.cnn import CnnModel
from networks.mlp import MLP
from networks.mlp_riga import MLP_RIGA
from networks.VAE_model import VAE
from networks.resnet import res_net18
from networks.resnet18_two_linear import ResNet18TwoLinear
from networks.VGG import VGG16
from torch.utils.data import Dataset

devices = 'cuda' if torch.cuda.is_available() else 'cpu'


class Util:
    @staticmethod
    def hard_th(matrix_g):
        return torch.nn.Threshold(0.5, 0)(matrix_g)

    @staticmethod
    def stack_act(extractor, x_key, config):
        x_fc = torch.cat([extractor(data.to(config["device"]))[config["layer_name"]].detach().cpu() for data, _ in
                          x_key], dim=0)
        return x_fc.cuda()

    @staticmethod
    def _get_act(dict_of_tensors):
        first_key = next(iter(dict_of_tensors))

        # Check if the first tensor has a shape of 4
        if len(dict_of_tensors[first_key].shape) == 4:
            concatenated_tensor_dict = dict_of_tensors[first_key].mean(dim=1)
        else:
            concatenated_tensor_dict = dict_of_tensors[first_key]

        # Reshape the first tensor
        concatenated_tensor_dict = concatenated_tensor_dict.view(concatenated_tensor_dict.shape[0], -1)

        # Loop through the dictionary starting from the second element
        for key in list(dict_of_tensors)[1:]:
            tmp = dict_of_tensors[key]

            # If the tensor has a shape of 4, take the mean along the second dimension
            if len(tmp.shape) == 4:
                tmp = tmp.mean(dim=1)

            # Reshape the tensor
            tmp = tmp.view(tmp.shape[0], -1)

            # Concatenate along the last dimension
            concatenated_tensor_dict = torch.cat((concatenated_tensor_dict, tmp), dim=1)

        return concatenated_tensor_dict

    @staticmethod
    def print_gpu_memory():
        """Print GPU memory in one compact line"""
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9  # GB
        cached = reserved - allocated
        print(f"GPU Memory → Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Cached: {cached:.2f}GB")

    @staticmethod
    def recalibrate_batchnorm(model, loader, device="cuda"):
        """Reset and recalibrate BatchNorm running statistics."""

        # Reset all BatchNorm running stats
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.reset_running_stats()
                m.momentum = None  # Use cumulative moving average

        # Recalibrate by running through data
        model.train()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                _ = model(images)

        return model

class Random:
    @staticmethod
    def get_rand_bits(size, a, b):
        return random.choices([a, b], k=size)

    @staticmethod
    def select_random_positions(shape, nb_samples):
        if isinstance(shape, int):
            pos_list = list(range(shape))
        else:
            pos = [range(s) for s in shape]
            pos_list = [list(p) for p in it.product(*pos)]
        sampled_list = random.sample(pos_list, nb_samples)
        return sampled_list

    @staticmethod
    def select_random_positions_percent(shape, percent):
        return Random.select_random_positions(shape, int(np.prod(shape) * percent))

    @staticmethod
    def generate_shuffled_list(width, high):
        lst = [list(p) for p in it.product(range(width), range(high))]
        return random.shuffle(lst)

    @staticmethod
    def generate_secret_matrix(width, high):
        # return torch.normal(mean=0, std=1, size=(width, high))
        return torch.randn(size=(width, high))
        # return 2.*torch.rand(size=(width, high)) - 1


class Database:

    @staticmethod
    def get_transforms(database):
        if database == "mnist":
            transform_train = transforms.Compose([
                transforms.ToTensor(), torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))])
            transform_test = transforms.Compose([
                transforms.ToTensor(), torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))])
        elif database == "cifar10" or database == "cifar100":
            transform_train = transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif database == "celeba":
            transform_train = transforms.Compose([
                transforms.Resize(64),  # Resize to 64x64
                transforms.CenterCrop(64),  # Center crop
                transforms.ToTensor(),  # Convert to tensor [0, 1]
                transforms.Normalize(  # Normalize to [-1, 1]
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
            transform_test = transforms.Compose([
                transforms.Resize(64),  # Resize to 64x64
                transforms.CenterCrop(64),  # Center crop
                transforms.ToTensor(),  # Convert to tensor [0, 1]
                transforms.Normalize(  # Normalize to [-1, 1]
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        elif database == "flower102" :
            transform_train = transforms.Compose([
                transforms.Resize(32,),
                transforms.RandomResizedCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32,),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        elif database == "stl10" :
            transform_train = transforms.Compose([
                transforms.Resize(32,),
                transforms.RandomResizedCrop(32),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32,),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            raise Exception("database doesn't exist")
        return transform_train, transform_test

    @staticmethod
    def get_dataset(database, transform_train, transform_test):
        if database == "mnist":
            train_dataset = torchvision.datasets.MNIST(root='./data',
                                                       train=True,
                                                       transform=transform_train,
                                                       download=True)
            test_dataset = torchvision.datasets.MNIST(root='./data',
                                                      train=False,
                                                      transform=transform_test)
        elif database == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                         train=True,
                                                         transform=transform_train,
                                                         download=True)
            test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                        train=False,
                                                        transform=transform_test)
        elif database == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                          train=True,
                                                          transform=transform_train,
                                                          download=True)
            test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                         train=False,
                                                         transform=transform_test)

        elif database == "flower102":
            train_dataset = torchvision.datasets.Flowers102(root='./data',
                                                          split='test',
                                                          transform=transform_train,
                                                          download=True)
            test_dataset = torchvision.datasets.Flowers102(root='./data',
                                                         split='train',
                                                         transform=transform_test) # use 'train' split for testing because test has 6149 image and train has 1020 images only
        elif database == "stl10":
            train_dataset = torchvision.datasets.STL10(root='./data',
                                                          split='train',
                                                          transform=transform_train,
                                                            download=True
                                                          )
            test_dataset = torchvision.datasets.STL10(root='./data',
                                                         split='test',
                                                         transform=transform_test)
        else:
            raise Exception("Unknown Database")

        return train_dataset, test_dataset

    @staticmethod
    def get_loaders(database, batch_size):
        """creat a new attack set with same dimension of training set"""
        transform_train, transform_test = Database.get_transforms(database)
        train_dataset, test_dataset = Database.get_dataset(database, transform_train, transform_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    @staticmethod
    def gen_dataset_loaders(config):
        train_loader, test_loader = Database.get_loaders(database=config["database"], batch_size=config["batch_size"])
        dataset = {"database": config["database"], "train_loader": train_loader, "test_loader": test_loader}
        torch.save(dataset, config["save_path"])
        print("train and test loaders have been created and saved successfully ")

    @staticmethod
    def load_dataset_loaders(config):
        dataset = torch.load(config["save_path"], weights_only=False)
        print("loading the following database... ", config["database"])
        return dataset["train_loader"], dataset["test_loader"]


class TrainModel:
    @staticmethod
    def fine_tune(init_model, train_loader, test_loader, config):

        model = deepcopy(init_model)
        model = model.to(config["device"])
        model.train()

        criterion = config["criterion"]
        optimizer = TrainModel.get_optimizer(model, config)
        scheduler = TrainModel.get_scheduler(optimizer, config)

        best_acc = 0
        best_model = deepcopy(model)
        best_model, best_acc, acc = TrainModel.check_acc(model=model, best_model=best_model, test_loader=test_loader,
                                                         config=config,
                                                         best_acc=best_acc, epoch=0)

        acc_list = []
        if config["show_acc_epoch"]:
            acc_list = [acc]

        for param in model.parameters():
            param.requires_grad = True

        # monitor = ActivationMonitor(model, layer_name="base.layer3.0.conv2")

        for epoch in range(config["epochs"]):
            train_loss = correct = total = 0
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # loss1 = criterion(outputs, targets)
                # loss_l2= sum(p.abs().sum() for p in model.parameters())
                # loss=loss1+config["lambda"]*loss_l2
                # back propagate the loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # update the progress bar
                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"
                                               f"/{total}]")
            # if epoch % 5 == 0:
            #     monitor.report(epoch)
            scheduler.step()
            best_model, best_acc, acc = TrainModel.check_acc(model=model, best_model=best_model,
                                                             test_loader=test_loader,
                                                             config=config,
                                                             best_acc=best_acc, epoch=epoch)

            if config["show_acc_epoch"]:
                acc_list.append(acc)
        # monitor.remove()
        if config["show_acc_epoch"]:
            # print("acc=", best_acc)
            return model, acc_list
        else:
            print("acc=", best_acc)
            return model

    @staticmethod
    def get_scheduler(optimizer, cf):
        if cf["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cf["milestones"], gamma=cf["gamma"])
        return scheduler

    @staticmethod
    def check_acc(model, best_model, test_loader, config, best_acc, epoch):
        acc = TrainModel.evaluate(model, test_loader, config)
        if acc > best_acc:
            model_info = {"model":model, "acc":acc, "epoch":epoch, "model_state_dict":model.state_dict()}
            os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
            torch.save(model_info, config["save_path"])
            # TrainModel.save_model(model, acc, epoch, config["save_path"])  # in config file we use save_path
            best_model = deepcopy(model)
            best_acc = acc
        return best_model, best_acc, acc

    @staticmethod
    def evaluate(init_model, test_loader, config):
        test_loss = correct = total = 0
        loop = tqdm(test_loader, leave=True)
        model = deepcopy(init_model)
        # model.eval()
        criterion = config["criterion"]
        device = config["device"]

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(device), targets.to(device)
                # targets = targets.type(torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # update the progress bar
                loop.set_description(f"Testing set ")
                loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=f"{100. * (correct / total)}",
                                 correct_total=f"[{correct}"f"/{total}]")
        acc = 100. * correct / total
        return acc

    @staticmethod
    def save_model(model, acc, epoch, path, supplementary=None):
        print('Saving model...')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'supplementary': supplementary
        }
        savedir = os.path.dirname(path)
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)
        torch.save(state, path)

    @staticmethod
    def load_model(path):
        print('Loading model...')
        state = torch.load(path, weights_only=False)
        return state

    @staticmethod
    def select_architecture(config_data, config_model):
        """select the suitable architecture"""
        """get train and test loader"""
        train_loader, test_loader = Database.get_loaders(database=config_data["database"],
                                                         batch_size=config_data["batch_size"])
        """load the trained model in the given path"""
        model = TrainModel.load_model(config_model["path_model"])['net']
        return model, train_loader, test_loader

    @staticmethod
    def get_model(architecture, device, n_classes=None):
        if architecture == "CNN":
            # ep50, bs =512, lr=0.001, Adam
            if n_classes is None:
                model = CnnModel().to(device)
            else:
                model = CnnModel(n_classes).to(device)
        elif architecture == "MLP":
            # ep50, bs =512, lr=0.01, opt=Adam
            model = MLP().to(device)
        # elif architecture == "ResNet18":
        #     model = res_net18().to(device)
        elif architecture == "ResNet18":
            model = ResNet18TwoLinear().to(device)
        elif architecture == "VGG16":
            model=VGG16(num_classes=n_classes).to(device)
        elif architecture == "MLP_RIGA":
            model = MLP_RIGA().to(device)
        elif architecture == "VAE":
            model = VAE(num_latent_dims=64, num_img_channels=3, max_num_filters=128,  device=device).to(device)
        else:
            raise Exception("architecture doesn't exist")
        return model

    @staticmethod
    def get_optimizer(model, config, parameters=None):
        if parameters is not None:
            return TrainModel._get_optimizer(config, list(model.parameters()) + list(parameters))
        else:
            return TrainModel._get_optimizer(config, model.parameters())

    @staticmethod
    def _get_optimizer(config, parameters):
        if config["opt"] == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=config["lr"], momentum=config["momentum"],
                                        weight_decay=config["wd"])
        elif config["opt"] == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["wd"])
        elif config["opt"] == "RMSprop":
            optimizer = torch.optim.RMSprop(parameters, lr=config["lr"], alpha=0.9, eps=1e-08,
                                            weight_decay=config["wd"],
                                            momentum=config["momentum"], centered=False)
        elif config["opt"] == "AdamW":
            optimizer = torch.optim.AdamW(parameters, lr=config["lr"])
        else:
            raise Exception("Unknown optimizer")
        return optimizer


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, input_tensor):
        return torch.randn(input_tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddWhiteSquareTransform:
    def __init__(self, square_size=20, start_x=30, start_y=40):
        self.square_size = square_size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, x):
        # Assuming x is a single image tensor of shape (C, H, W)
        C, H, W = x.size()
        white_square = torch.ones((C, self.square_size, self.square_size))
        if self.start_x + self.square_size <= H and self.start_y + self.square_size <= W:
            x[:, self.start_x:self.start_x + self.square_size,
            self.start_y:self.start_y + self.square_size] = 1  # white_square
        else:
            raise ValueError("Square position and size exceed image dimensions.")
        return x


class CustomTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform_list=None):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.x_tensor)


class ActivationMonitor:
    """Monitor activations during training"""

    def __init__(self, model, layer_name):
        self.layer_name = layer_name
        self.activations = []
        self.hook_handle = None

        # Register hook
        for name, module in model.named_modules():
            if name == layer_name:
                self.hook_handle = module.register_forward_hook(self._hook)
                break

    def _hook(self, module, input, output):
        """Capture activation statistics"""
        with torch.no_grad():
            stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'zero_ratio': (output == 0).float().mean().item(),
                'near_zero_ratio': (output.abs() < 1e-6).float().mean().item()
            }
            self.activations.append(stats)

    def report(self, epoch=None):
        """Print activation report"""
        if not self.activations:
            print("No activations recorded")
            return

        recent = self.activations[-1]
        prefix = f"Epoch {epoch} - " if epoch is not None else ""

        print(f"{prefix}{self.layer_name}:")
        print(f"  Mean: {recent['mean']:.6f}, Std: {recent['std']:.6f}")
        print(f"  Range: [{recent['min']:.6f}, {recent['max']:.6f}]")
        print(f"  Zero ratio: {recent['zero_ratio']:.2%}")

        if recent['zero_ratio'] > 0.9:
            print(f"  ⚠️  WARNING: {recent['zero_ratio']:.1%} of activations are zero!")
        elif recent['mean'] == 0 and recent['std'] == 0:
            print(f"  ⚠️  WARNING: Layer appears to be dead (all zeros)!")

    def remove(self):
        """Remove hook"""
        if self.hook_handle:
            self.hook_handle.remove()


class AddWatermarkPatternTransform:
    """class CustomTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform_list=None):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.x_tensor)
    Ajoute un pattern de tatouage haute fréquence à une image.
    Compatible avec torchvision.transforms.Compose
    """

    def __init__(self,
                 pattern_size=20,
                 start_x=30,
                 start_y=40,
                 intensity=0.1,
                 pattern_type='high_freq',
                 seed=None):
        """
        Args:
            pattern_size: Taille du pattern carré (ex: 20x20)
            start_x: Position X de départ du pattern
            start_y: Position Y de départ du pattern
            intensity: Intensité de la perturbation (0.05-0.2 recommandé)
            pattern_type: 'high_freq', 'checkerboard', ou 'random'
            seed: Graine aléatoire pour reproductibilité
        """
        self.pattern_size = pattern_size
        self.start_x = start_x
        self.start_y = start_y
        self.intensity = intensity
        self.pattern_type = pattern_type
        self.seed = seed

        # Générer le pattern une seule fois (cache)
        self.pattern = self._generate_pattern()

    def _generate_pattern(self):
        """Génère le pattern de tatouage haute fréquence"""
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if self.pattern_type == 'high_freq':
            # Pattern binaire [-1, 1] aléatoire (optimal pour robustesse)
            pattern = np.random.choice([-1, 1],
                                       size=(self.pattern_size, self.pattern_size))
            pattern = torch.tensor(pattern, dtype=torch.float32)

        elif self.pattern_type == 'checkerboard':
            # Pattern damier (très haute fréquence, régulier)
            pattern = torch.zeros(self.pattern_size, self.pattern_size)
            for i in range(self.pattern_size):
                for j in range(self.pattern_size):
                    pattern[i, j] = 1 if (i + j) % 2 == 0 else -1

        elif self.pattern_type == 'random':
            # Pattern gaussien (moins robuste mais plus discret)
            pattern = torch.randn(self.pattern_size, self.pattern_size)
            pattern = torch.clamp(pattern, -1, 1)

        else:
            raise ValueError(f"pattern_type inconnu: {self.pattern_type}")

        return pattern * self.intensity

    def __call__(self, x):
        """
        Applique le pattern de tatouage à l'image

        Args:
            x: Tensor de forme (C, H, W) - image normalisée

        Returns:
            Tensor de forme (C, H, W) avec pattern ajouté
        """
        C, H, W = x.size()

        # Vérifier que le pattern rentre dans l'image
        if self.start_x + self.pattern_size > H or self.start_y + self.pattern_size > W:
            raise ValueError(
                f"Pattern ({self.pattern_size}x{self.pattern_size}) à position "
                f"({self.start_x}, {self.start_y}) dépasse les dimensions de l'image ({H}x{W})"
            )

        # Cloner pour ne pas modifier l'original
        x_watermarked = x.clone()

        # Appliquer le pattern sur tous les canaux
        for c in range(C):
            x_watermarked[c,
            self.start_x:self.start_x + self.pattern_size,
            self.start_y:self.start_y + self.pattern_size] = self.pattern

        # Clip pour garder les valeurs valides
        # Si normalisé [-1, 1]: clamp(-1, 1)
        # Si normalisé [0, 1]: clamp(0, 1)
        # x_watermarked = torch.clamp(x_watermarked, -1, 1)  # Ajuster selon votre normalisation

        return x_watermarked

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"pattern_size={self.pattern_size}, "
                f"position=({self.start_x}, {self.start_y}), "
                f"intensity={self.intensity}, "
                f"type='{self.pattern_type}')")



