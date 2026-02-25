from copy import deepcopy
import random
import torch
from torch import nn, optim
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from watermark.diction import Diction
from networks.linear_mod import ProjMod

# Check Device configuration
from util.util import TrainModel, Random, CustomTensorDataset, AddGaussianNoise, AddWhiteSquareTransform, Util

from watermark.watermark  import Watermark


def ambiguity_attack(fake_orig_model, wat_model, train_loader, test_loader ,config)  :
    diction=Diction()
    watermark= diction.keygen(config["watermark_size"])
    watermark_rd = diction.keygen(config["watermark_size"])

    wat_model = deepcopy(wat_model)
    orig_model_copy = deepcopy(fake_orig_model)

    # Get the activation layer of the original wat_model and make sure that its parameters are not trainable
    extractor_orig = create_feature_extractor(orig_model_copy, config["layer_name"])
    # Get the activation layer of the target wat_model
    extractor_wat = create_feature_extractor(wat_model, config["layer_name"])
    # show the graph of the wat_model
    print("the wat_model graph :=> ", get_graph_node_names(orig_model_copy)[0])

    for param in extractor_wat.parameters():
        param.requires_grad = False

        # Generate the trigger set based on a Latent space
    x_key, y_key = next(iter(train_loader))

    transform_train = transforms.Compose([
        # transforms.RandomCrop(size=32, padding=4),
        # transforms.RandomHorizontalFlip(),
        AddGaussianNoise(config["mean"], config["std"]),
        AddWhiteSquareTransform(square_size=config["square_size"], start_x=config["start_x"],
                                start_y=config["start_y"]),
    ])

    dataset_key = CustomTensorDataset(x_key, y_key, transform_list=transform_train)
    key_loader = DataLoader(dataset=dataset_key, batch_size=config["batch_size"], shuffle=False)

    # Get the number of features of the layer
    # Get trigger set images from the key loader
    x_key, _ = next(iter(key_loader))
    x_key = x_key.cuda()  # 128,3,32,32

    # Get the activation maps of all involved layers
    tmp_data = extractor_orig(x_key)
    tmp_data = Util._get_act(tmp_data)  # 128*100 10*10 conv4

    # Compute the number of features that will curry the watermark
    n_features_layer = len(tmp_data[0])  # 100 conv4
    config["n_features"] = int(n_features_layer * config["n_features"])
    print("n_features_layer: ", n_features_layer, "n_features selected: ", config["n_features"])
    # Get their indices
    indices = random.choices(range(n_features_layer), k=config["n_features"])
    print("indices: ", len(indices))

    # Instance the linear mod
    proj_mod = ProjMod(config).to(config["device"])

    # The parameters of training
    criterion = config["criterion"]
    # Optimizers
    optimizer_proj = optim.AdamW(proj_mod.parameters(), lr=config["lr_proj"],
                                 weight_decay=config["wd_proj"])  # 1e-4
    optimizer_model = optim.AdamW(orig_model_copy.parameters(), lr=config["lr"], weight_decay=config["wd"])
    # Schedulers
    proj_scheduler = TrainModel.get_scheduler(optimizer_proj, config)
    model_scheduler = TrainModel.get_scheduler(optimizer_model, config)

    # Init the BER and the losses
    ber_ = ber = l_proj = 1
    dict_model={}

    # Start the training
    for epoch in range(config["epochs"]):
        embed_loss = train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)
        proj_mod.train(True)
        orig_model_copy.train(True)
        extractor_wat.train(True)

        for batch_idx, (x_train, y_train) in enumerate(loop):
            # Get data from the trigger set
            x_key, _ = next(iter(key_loader))
            x_key = x_key.cuda()

            optimizer_proj.zero_grad()
            optimizer_model.zero_grad()

            # Train the Proj_ Model
            # Get activation maps of the watermarked wat_model
            act_wat = extractor_wat(x_key)
            act_wat = Util._get_act(act_wat)
            act_wat = act_wat[:, indices]

            # Get activation maps of the original wat_model
            act_orig = extractor_orig(x_key)
            act_orig = Util._get_act(act_orig)
            act_orig = act_orig[:, indices]

            # Get the output of the projection wat_model
            watermark_out = proj_mod(act_wat)
            watermark_orig_out = proj_mod(act_orig.detach())

            l_wat = BCELoss(reduction='mean')(watermark_out, watermark.repeat(len(watermark_out), 1).cuda())
            l_wat_orig = BCELoss(reduction='mean')(watermark_orig_out,
                                                   watermark_rd.repeat(len(watermark_out), 1).cuda())
            l_proj = config["lambda_proj"] * (l_wat_orig + l_wat)

            # Train the target wat_model with the constraint  on the AM of the target wat_model
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            y_train = y_train.type(torch.long)

            y_pred = orig_model_copy(x_train)
            l_main_task = criterion(y_pred, y_train)

            l_model = l_main_task + config["lambda_proj"] * l_wat_orig


            l_proj.backward(retain_graph=True)
            l_model.backward()

            optimizer_proj.step()
            optimizer_model.step()

            train_loss += l_main_task.item()
            embed_loss += l_proj.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            # update the progress bar
            ber = diction._get_ber(watermark_out, watermark.repeat(len(watermark_out), 1))
            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(l_main_task=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", l_proj=f"{embed_loss / (batch_idx + 1):1.4f}",
                             ber=f"{ber:1.3f}",
                             l_wat=f"{l_wat:1.6f}", l_wat_orig=f"{l_wat_orig:1.6f}"
                             )

        proj_scheduler.step()
        model_scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():


                act_wat = extractor_wat(x_key.cuda())
                act_wat = Util._get_act(act_wat)
                act_wat = act_wat[:, indices]

                _, ber_ = diction._extract(act_wat, proj_mod, watermark.cuda())  # proj_mod.fc.bias
                acc = TrainModel.evaluate(orig_model_copy, test_loader, config)
                print(
                    f"epoch:{epoch}---l_global: {l_proj.item():1.7f}---ber_mean: {ber_:1.3f}"
                    f"---param_var_loss: "
                    f"{l_proj:1.4f}---acc: {acc}")

            if ber_ == 0:
                print("saving... watermarked wat_model ")
                supplementary = {'wat_model': wat_model, 'matrix_a': proj_mod, 'watermark': watermark,
                                 'watermark_r': watermark_rd,
                                 'x_key': x_key, 'y_key': y_key, 'ber': ber,
                                 "layer_name": config["layer_name"], "indices": indices}
                dict_model=supplementary
                print("dict model", dict_model.keys())
                diction.save(path=config['save_path_wat'], supplementary=supplementary)
                # TrainModel.save_model(deepcopy(wat_model), acc, epoch, config['save_path'], supplementary)
                print("wat_model saved!")
                print("saving...  fake fine tuned model ")
                supplementary = {'model': orig_model_copy}
                diction.save(path=config['save_path_orig'], supplementary=supplementary)
                break

    print("dict model2", dict_model.keys())
    return orig_model_copy, dict_model
