import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ******************* CF Mnist *******************
database = "mnist"
batch_size = 128

cf_mnist_data = {"batch_size": batch_size, "database": database, "device": device, "channels": 1,"n_classes":10,
                 "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}

# ******************* CF Cifar10 *******************
database = "cifar10"
batch_size = 128 #32 for vgg16 and 128 for resnet18, cnn

cf_cifar10_data = {"batch_size": batch_size, "database": database, "device": device ,"channels": 3,"n_classes":10, "height":32,"width":32,
                   "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}

# ******************* CF Cifar100 *******************
database = "cifar100"
batch_size = 128

cf_cifar100_data = {"batch_size": batch_size, "database": database, "device": device, "channels": 3,"n_classes":100,
                   "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}


# ******************* CF flower102 *******************
database = "flower102"
batch_size = 128

cf_flower102_data = {"batch_size": batch_size, "database": database, "device": device, "channels": 3, "n_classes":102,
                   "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}


# ******************* CF flower102 *******************
database = "stl10"
batch_size = 128

cf_stl10_data = {"batch_size": batch_size, "database": database, "device": device, "channels": 3, "n_classes":10,
                   "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}