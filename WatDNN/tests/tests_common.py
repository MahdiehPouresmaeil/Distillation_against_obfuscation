import torch
from torch import nn, optim
from torch.ao.quantization.pt2e.export_utils import model_is_exported
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from attacks.pruning import pruning
from attacks.dummy_neurons import neuron_clique, neuron_split
from attacks.distillation import train_student, train_student_hufu
from networks.cnn import CnnModel
from networks.piadetector import PiaDetector
from util.metric import Metric
from util.util import TrainModel, Database, CustomTensorDataset, Util
from networks.hufu_net import HufuNet
from util.hufu_func import Hufu_func

from networks.cnn import CnnModel
from configs.cf_data .cf_data import  cf_cifar10_data
from configs.cf_train.cf_cnn import cf_cnn_dict
import math
from attacks.ambiguity import ambiguity_attack
import torch.backends.cudnn
torch.backends.cudnn.enabled = False
from networks.VAE_model import VAE


class Tests:
    def __init__(self, method: str, model: str):
        self.model = model
        self.method = method
        self.watermark = self.get_watermark(method)


    def embedding(self, config_embed, config_data):

        # Get the original model

        if self.model in {"VAE"}:
            init_model = VAE(num_latent_dims=config_embed["LATENT_DIMS"], num_img_channels=config_data["channels"], max_num_filters=config_embed["MAX_FILTERS"],
                             device=config_embed["device"])
            state_dict = torch.load(config_embed["path_model"], weights_only=False)
            init_model.load_state_dict(state_dict)
        else:
            init_model = torch.load(config_embed["path_model"], weights_only=False)['model']
            print(config_embed["path_model"])
        # Get the training and testing data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # evaluate the original model
        print(f"evaluate the original model before watermarking with {self.method}")
        acc=TrainModel.evaluate(init_model, test_loader, config_embed)
        print("ACC initial model = ", acc)
        # embed the watermark
        model_wat, ber = self.watermark.embed(init_model, test_loader, train_loader, config_embed)
        print(f"evaluate the watermarked model with {self.method}")
        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
        return acc, ber

    def fine_tune_attack(self, config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # Get model
        init_model = torch.load(config_embed["path_model"],weights_only=False)['model']
        # init_model.eval()
        # load the watermarked model
        dict_model = torch.load(config_embed['save_path'],weights_only=False)

        print(config_embed['save_path'])
        if config_embed['method_wat'] in {"diction" ,"diction2"}:
            model_wat = dict_model["wat_model"]
        else:
            model_wat = dict_model["model"]

        # fine tune the watermarked model
        print("Check the accuracy of the watermarked model...")
        accu=TrainModel.evaluate(model_wat, test_loader, config_attack)
        print(f"accuracy of the watermarked model: {accu:.2f}%")
        print("Compute the BER from the original model (non watermarked)...")

        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"} :
            _, ber = self.watermark.extract(init_model, dict_model, config_embed)
        else:
            _, ber = self.watermark.extract(init_model, dict_model)
        print("BER = ", ber)
        print("Compute the BER from the watermarked model ...")
        if config_embed['method_wat']  in {"greedy", "Laplacian_dis"} :
            _, ber = self.watermark.extract( model_wat, dict_model, config_embed)
        else:
            _, ber = self.watermark.extract( model_wat, dict_model)

        print("BER = ", ber)
        print("Start fine-tuning...")
        results_acc = []
        results_ber = []
        epochs = [config_attack["epochs"]*i for i in range(1, 4)]
        print(epochs)
        # epochs = [10]


        for ep in epochs:
            print("ep", ep)
            model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
            # check the accuracy and BER
            if config_embed['method_wat'] in {"greedy", "Laplacian_dis"} :
                _, ber = self.watermark.extract(model_wat, dict_model,config_embed)
            else:
                _, ber = self.watermark.extract(model_wat, dict_model)
            acc = TrainModel.evaluate(model_wat, test_loader, config_attack)
            print(f"ACC and BER after finetuning {ep}")
            print("BER = ", ber, "ACC = ", acc)
            results_acc.append(acc)
            results_ber.append(ber)
        print("epochs = ", epochs)
        print("results_acc = ", results_acc)
        print("results_ber = ", results_ber)

    def pruning_attack(self, config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = torch.load(config_embed['save_path'], weights_only=False)

        if config_embed['method_wat'] in {"diction", "diction2"}:
            model_wat = dict_model["wat_model"]
        else:
            model_wat = dict_model["model"]

        print(dict_model["ber"])
        print(dict_model["layer_name"])
        # fine tune the watermarked model
        print("First check before pruning")
        acc=TrainModel.evaluate(model_wat, test_loader, config_attack)
        print("acc watermarked model before pruning = ", acc)
        results_acc = []
        results_ber = []
        pruning_rate = [x / 10 for x in range(10)] + [0.95, 0.99, 1.]

        for rate in pruning_rate:
            model = pruning(model_wat, rate)
            print("evaluate the model after pruning of amount", rate)
            results_acc.append(TrainModel.evaluate(model, test_loader, config_attack))

            if config_embed['method_wat']in {"greedy", "Laplacian_dis"} :
                results_ber.append(self.watermark.extract(model, dict_model, config_attack)[1])
            else:
                results_ber.append(float(self.watermark.extract(model, dict_model)[1]))


        print("pruning_rate = ", pruning_rate)
        print("results_acc = ", results_acc)
        print("results_ber = ", results_ber)

    def overwriting_attack(self, config_embed, config_attack, config_data):
        # Get model
        init_model = torch.load(config_embed["path_model"], weights_only=False)['model']
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = torch.load(config_embed['save_path'], weights_only=False)
        if config_embed['method_wat'] in {"diction", "diction2"}:
            model_wat = dict_model["wat_model"]
        else:
            model_wat = dict_model["model"]
        # evaluate the model
        print("Evaluate the model before watermarking")
        acc=TrainModel.evaluate(model_wat, test_loader, config_embed)
        print("ACC = ", acc)

        # print(dict_model)
        print("Check BER of watermarked model")
        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"} :
            _, ber = self.watermark.extract(model_wat, dict_model, config_embed)
        else:
            _, ber = self.watermark.extract(model_wat, dict_model)
        print("BER = ", ber)
        # embed the watermark with the overwriting attack
        model_attacked, ber = self.watermark.embed(model_wat, test_loader, train_loader, config_attack)
        # evaluate the attacked model
        dict_model_attacked = torch.load(config_attack['save_path'], weights_only=False)
        if config_embed['method_wat'] in {"diction","diction2"}:
            model_wat = dict_model["wat_model"]
        else:
            model_wat = dict_model["model"]

        acc = TrainModel.evaluate(model_attacked, test_loader, config_attack)

        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"} :
            b_ext_1, ber_1 = self.watermark.extract(model_attacked, dict_model_attacked,config_attack)
            b_ext_2, ber_2 = self.watermark.extract(model_attacked, dict_model,config_attack)
            b_ext_3, ber_3 = self.watermark.extract(init_model, dict_model_attacked,config_attack)
            b_ext_4, ber_4 = self.watermark.extract(init_model, dict_model,config_attack)
            b_ext_5, ber_5 = self.watermark.extract(model_wat, dict_model_attacked,config_attack)
            b_ext_6, ber_6 = self.watermark.extract(model_wat, dict_model,config_attack)
        else:
            b_ext_1, ber_1 = self.watermark.extract(model_attacked, dict_model_attacked)
            b_ext_2, ber_2 = self.watermark.extract(model_attacked, dict_model)
            b_ext_3, ber_3 = self.watermark.extract(init_model, dict_model_attacked)
            b_ext_4, ber_4 = self.watermark.extract(init_model, dict_model)
            b_ext_5, ber_5 = self.watermark.extract(model_wat, dict_model_attacked)
            b_ext_6, ber_6 = self.watermark.extract(model_wat, dict_model)

        print(f"BER_1 (attacked model with overwrite projection model): {ber_1}")
        print(f"BER_2 (attacked model with watermark projection model): {ber_2}")
        print(f"BER_3 (original model with overwrite projection model): {ber_3}")
        print(f"BER_4 (original model with watermark projection model): {ber_4}")
        print(f"BER_5 (watermarked model with overwrite projection model): {ber_5}")
        print(f"BER_6 (watermarked model with watermark projection model): {ber_6}")



        return acc, ber

    def overwriting_attack_hufu(self, config_embed, config_attack, config_data):
        # Get model
        init_model = deepcopy(torch.load(config_embed["path_model"], weights_only=False)['model'])
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = torch.load(config_embed['save_path'], weights_only=False)
        model_wat= deepcopy(dict_model["supplementary"]["model"].to(config_embed["device"]))
        # evaluate the model
        print("Evaluate the model before watermarking")
        acc=TrainModel.evaluate(deepcopy(model_wat), test_loader, config_embed)
        print("ACC = ", acc)
        selected_indexes_original=dict_model["supplementary"]["selected_indexes"]



        # Get data
        train_loader_hufu, val_loader_hufu, test_loader_hufu = Hufu_func.create_data_loaders_hufu(
            batch_size=config_embed['batch_size_hufu'],
            validation_split=config_embed['validation_split']
        )

        # load the hufu model
        dict_model_hufu = torch.load(config_embed['save_path_hufu_finetune'], weights_only=False)
        print("dict_model_hufu", dict_model_hufu.keys())
        init_hufu_model = deepcopy(dict_model_hufu["supplementary"]["model"].to(config_embed["device"]))


        # print(dict_model)
        print("Check MSE of watermarked model")

        mse_before, mse_after, mse_non_wm=self.watermark.extract(deepcopy(model_wat), deepcopy(init_hufu_model), selected_indexes_original, train_loader_hufu, config_embed)
        print("MSE = ", mse_after)
        print("MSE hufu = ", dict_model_hufu["supplementary"]['MSE'])






        # embed the watermark with the overwriting attack
        model_attacked,selected_indexes_att = self.watermark.embed(model_wat, test_loader, train_loader, config_attack)
        # evaluate the attacked model
        # dict_model_attacked = torch.load(config_attack['save_path'], weights_only=False)
        # model_wat =deepcopy(dict_model["supplementary"]["model"].to(config_embed["device"]))

        acc = TrainModel.evaluate(model_attacked, test_loader, config_attack)

        # load the hufu model
        dict_model_hufu_attack = torch.load(config_attack['save_path_hufu_finetune'], weights_only=False)
        hufu_model_attack = deepcopy(dict_model_hufu_attack["supplementary"]["model"].to(config_attack["device"]))


        _, mse_after1, mse_non_wm1=self.watermark.extract(deepcopy(model_attacked), hufu_model_attack, selected_indexes_att,
                                               train_loader_hufu, config_attack)
        _, mse_after2, _ = self.watermark.extract(deepcopy(model_attacked), init_hufu_model, selected_indexes_original,
                                                 train_loader_hufu, config_attack)
        _, mse_after3, _ = self.watermark.extract(deepcopy(model_wat), hufu_model_attack, selected_indexes_att,
                                                 train_loader_hufu, config_attack)
        _, mse_after4, mse_non_wm2 = self.watermark.extract(deepcopy(model_wat), init_hufu_model, selected_indexes_original,
                                                 train_loader_hufu, config_attack)



        print(f"BER_1 (attacked model with overwrite projection model): {mse_after1}")
        print(f"BER_2 (attacked model with watermark projection model): {mse_after2}")
        print(f"BER_3 (original model with overwrite projection model): {mse_non_wm1}")
        print(f"BER_4 (original model with watermark projection model): {mse_non_wm2}")
        print(f"BER_5 (watermarked model with overwrite projection model): {mse_after3}")
        print(f"BER_6 (watermarked model with watermark projection model): {mse_after4}")



        return acc, _

    def show_weights_distribution(self, config_embed, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # Get model
        model_init = torch.load(config_embed["path_model"])['model']

        # Layer to inspect
        if isinstance(config_embed["layer_name"], list):
            layer_name = config_embed["layer_name"][-1]
        else:
            layer_name = config_embed["layer_name"].replace(".weight", "")

        # evaluate the original model
        print("evaluate the original model")
        TrainModel.evaluate(model_init, test_loader, config_embed)

        # load the watermarked model
        model_dict = torch.load(config_embed['save_path'])
        model_wat = model_dict["model"]

        # get trigger set
        # x_key, _ = next(iter(model_dict["x_key"]))

        # evaluate the original model
        print("evaluate the watermarked model")
        print("layer_name", layer_name)
        if layer_name == "linear":  # in the case of resnet18
            layer_name = "view"
        TrainModel.evaluate(model_wat, test_loader, config_embed)
        extractor_init = create_feature_extractor(model_init, [layer_name])
        extractor_wat = create_feature_extractor(model_wat, [layer_name])

        x_train, _ = next(iter(train_loader))  # x
        # x_train = x_key

        # Get activation distributions
        act_init = extractor_init(x_train.cuda())[layer_name]
        act_wat = extractor_wat(x_train.cuda())[layer_name]
        # Compute the mean of activation maps across the batch dimension
        act_init = torch.mean(act_init, dim=0)
        act_wat = torch.mean(act_wat, dim=0)

        # Flatten the activation maps to simplify histogram computation
        act_init_flat = torch.flatten(act_init.data).cpu().numpy()
        act_wat_flat = torch.flatten(act_wat.data).cpu().numpy()

        # Compute the min and max for dynamic binning
        min_act = min(act_init_flat.min(), act_wat_flat.min())
        max_act = max(act_init_flat.max(), act_wat_flat.max())

        # Generate bins dynamically from min to max with steps
        bin_size = 1  # adjust bin size as needed
        bins = np.arange(min_act, max_act + bin_size, bin_size)

        # Compute activation distribution stats
        act_init_mean, act_init_std = act_init_flat.mean(), act_init_flat.std()
        act_wat_mean, act_wat_std = act_wat_flat.mean(), act_wat_flat.std()

        # Plot distributions
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 11.25))

        # Init activations hist
        ax1.hist(act_init_flat, bins=bins, align='mid', edgecolor='red')
        ax1.set(xlabel='Bins', ylabel='Frequency', title='non watermarked activation maps')
        ax1.text(0.5, 0.9, f"mean : {act_init_mean:.2f} \nstd : {act_init_std:.2f}",
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax1.transAxes)

        # Watermarked activations hist
        ax2.hist(act_wat_flat, bins=bins, align='mid', edgecolor='red')
        ax2.set(xlabel='Bins', title='watermarked activation maps')
        ax2.text(0.5, 0.9, f"mean : {act_wat_mean:.2f} \nstd : {act_wat_std:.2f}",
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax2.transAxes)

# Plot model parameter distributions
        for (name_init, param_init), (name, param) in zip(model_init.named_parameters(), model_wat.named_parameters()):
            if name == layer_name + '.weight' or name == "linear.weight":  # only for ResNet18
                # Flatten the weight tensors for histogram plotting
                weights_init = torch.flatten(param_init.data).cpu().numpy()
                weights_wat = torch.flatten(param.data).cpu().numpy()

                # Calculate statistics
                weights_init_mean, weights_init_std = weights_init.mean(), weights_init.std()
                weights_wat_mean, weights_wat_std = weights_wat.mean(), weights_wat.std()

                # Update global min and max for init and wat models
                min_weights = min(weights_init.min(), weights_wat.min())
                max_weights = max(weights_init.max(), weights_wat.max())

                # Compute bins based on overall min and max
                bins = list(np.arange(min_weights, max_weights, 0.1))

                # Histogram for non-watermarked weights
                ax3.hist(weights_init, bins=bins, align='mid', edgecolor='red')
                ax3.set(xlabel='Bins', ylabel='Frequency', title='Non-watermarked weights')
                ax3.text(0.5, 0.9, f"Mean: {weights_init_mean:.2f}\nStd: {weights_init_std:.2f}",
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax3.transAxes)

                # Watermarked
                ax4.hist(weights_wat, bins=bins, align='mid', edgecolor='red')
                ax4.set(xlabel='Bins', ylabel='Frequency', title='Watermarked weights')
                ax4.text(0.5, 0.9, f"Mean: {weights_wat_mean:.2f}\nStd: {weights_wat_std:.2f}",
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax4.transAxes)
                break

        savedir = os.path.join("results/weights", self.model)
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{self.method}_{self.model}.png"))
        plt.show()

    def dummy_neurons_attack(self, config_embed, config_attack, config_data):
        train_loader, test_loader = Database.load_dataset_loaders(config_data)

        dict_model = torch.load(config_attack['path_model'], weights_only=False)
        print(config_attack['path_model'])
        print(dict_model.keys())



        if config_embed['method_wat'] in {"diction"}:
            model_wat = dict_model["wat_model"]
        else:
            model_wat = dict_model["model"]

        print(model_wat)


        print("Check the accuracy of the watermarked model")
        acc=TrainModel.evaluate(model_wat, test_loader, config_embed)
        print(f"accuracy of the watermarked model: {acc:.2f}%")

        _, ber = self.watermark.extract(model_wat, dict_model)
        print("BER from the watermarked model  == ", ber)

        print("Original teacher model:")
        linear_layer_indices = []

        for i, (name, layer) in enumerate(model_wat.named_modules()):
            if isinstance(layer, (nn.Conv2d)):

                # print(f"{i}.  layer '{name}': {layer.in_features} -> {layer.out_features}")
                print(f"{i}.  layer '{name}': {layer.in_channels} -> {layer.out_channels}")

                linear_layer_indices.append((i, name))
            elif isinstance(layer, (nn.Linear)):

                # print(f"{i}.  layer '{name}': {layer.in_features} -> {layer.out_features}")
                print(f"{i}.  layer '{name}': {layer.in_features} -> {layer.out_features}")

                linear_layer_indices.append((i, name))

        print(f"Positions of nn.Linear layers: {linear_layer_indices}")

        if config_attack["attack_type"] == "neuron_clique":
            # # Test NeuronClique
            print("Testing NeuronClique:")
            print("-" * 30)
            attacked_model = neuron_clique(model=model_wat.to(config_attack["device"]),
                                         layer_name=config_attack["layer_name"], num_dummy=config_attack["num_dummy"])
            #
            for i, layer in enumerate(attacked_model.modules()):
                if isinstance(layer, nn.Linear):
                    print(f"{i}. Linear layer: {layer.in_features} -> {layer.out_features}")
                    linear_layer_indices.append(i)
        else:
            # # Test NeuronClique
            print("Testing NeuronSPLIT:")
            print("-" * 30)
            # new_model = neuron_split(model, layer_name="encoder.mlp.fc1", neuron_idx=3, num_splits=4)
            attacked_model = neuron_split(model=model_wat.to(config_attack["device"]),
                                          layer_name=config_attack["layer_name"], neuron_idx=config_attack["neuron_idx"]
                                          , num_splits=config_attack["num_splits"])
            #

            for i, layer in enumerate(attacked_model.modules()):
                if isinstance(layer, nn.Linear):
                    print(f"{i}. Linear layer: {layer.in_features} -> {layer.out_features}")
                    linear_layer_indices.append(i)

        # I need to save the model
        savedir = os.path.dirname(config_attack['save_path'])
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)
        dict_model["wat_model"] = attacked_model
        torch.save(dict_model, config_attack['save_path'])
        print("Attacked model with " + config_attack["attack_type"] + " performance")
        TrainModel.evaluate(attacked_model.to(config_attack["device"]), test_loader, config_embed)







    def distillation(self, config_embed, config_attack, config_data):
        # start by laoding the data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # I need to load the watermarked model for having the watermarking parameters
        # supp_wat_model = torch.load(config_attack['path_model'], weights_only=False)
        # print(f"distillation from {config_attack['path_model']}")
        # print(supp_wat_model.keys())


        #======load the watermarked model for distillation=======
        supp_wat_model = torch.load(config_embed['save_path'], weights_only=False)
        print(f"distillation from {config_embed['save_path']}")
        print(supp_wat_model.keys())

        # ======load the watermarked Dummy Neuron attacked model for distillation=======
        # supp_wat_model = torch.load(config_attack['path_model'], weights_only=False)
        # print(f"distillation from {config_attack['path_model']}")
        # print(supp_wat_model.keys())



        supp_non_model = torch.load(config_embed['path_model'], weights_only=False)['model']
        print(f"original non-watermarked model is  {config_embed['path_model']}")

        # supp_wat_model = torch.load(config_embed['save_path'], weights_only=False)
        # print(f"distillation from {config_embed['save_path']}")
        # #

        #for fine_tuned model
        # fine_funed_wat_model = torch.load(config_attack['path_fine_tuned'], weights_only=False)['model']
        # print(f"distillation from {config_attack['path_fine_tuned']}")


        #for distillation after ambiguity attack
        # supp_ambiguity_wat_model = torch.load(config_attack['path_ambiguity'], weights_only=False)
        # print(f"distillation from {config_attack['path_ambiguity']}")

        # for false alarm check ; read another non watermarked model
        # path='/home/latim/PycharmProjects/WatDNN/results/trained_models/vgg16/_dbcifar10_ep200_bs128.pth'
        # supp_non_wm_model = torch.load(path, weights_only=False)['model']
        # print(f"distillation from {path}")




        if config_embed['method_wat'] in {"diction","diction2"}:
            model_wat = supp_wat_model["wat_model"]
            model_wat = model_wat.to(config_attack["device"])

        else:
            model_wat = supp_wat_model["model"]

        #if you want to distillate from the fine tuned model
        # model_wat=deepcopy(fine_funed_wat_model)

        #distll from non watermarked model( false alarm check)
        # model_wat=supp_non_model
        # model_wat=supp_non_wm_model
        # model_wat=CnnModel().to(config_embed["device"])

        model_wat = Util.recalibrate_batchnorm(model_wat, train_loader, config_attack["device"])
        print("Check the accuracy of the watermarked model")
        model_wat.eval()

        with torch.no_grad():
            acc=TrainModel.evaluate(model_wat, test_loader, config_embed)
            print(f"accuracy of the watermarked model: {acc:.2f}%")


        acc = TrainModel.evaluate(supp_non_model, test_loader, config_embed)
        print(f"accuracy of the original model: {acc:.2f}%")

        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"}:
            _, ber = self.watermark.extract(model_wat, supp_wat_model, config_embed)
            _, bernon = self.watermark.extract(supp_non_model, supp_wat_model,config_embed)
        else:
            _, ber = self.watermark.extract(model_wat, supp_wat_model)
            _, bernon = self.watermark.extract(supp_non_model, supp_wat_model)
            #for ambiguity attacked model
            # _, ber_attacked = self.watermark.extract(model_wat, supp_ambiguity_wat_model)
        print("BER from the watermarked model (non attacked) == ", ber)
        print("BER from the original model (non attacked) == ", bernon)
        # for ambiguity attacked model
        # print("BER from the watermarked model ( ambiguty attacked) == ", ber_attacked)
        # I need to load the teacher model that just has been attacked with the dummy neurons and called the teacher
        # teacher = torch.load(config_attack['path_model']).to(config_embed["device"])
        teacher=deepcopy(model_wat)


        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"}:
            _, ber = self.watermark.extract(teacher, supp_wat_model,config_embed)
        else:
            _, ber = self.watermark.extract(teacher, supp_wat_model)

        print("BER of the teacher == ", ber)

        # I need to load the student model not watermarked and check if it s not watermarked
        # student = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        # print("student is the initialied model from scratch ")

        # student = torch.load(config_embed["path_model"],weights_only=False)['model']
        student=deepcopy(supp_non_model)
        # student=deepcopy(supp_non_wm_model)
        ##if you want to load the original model as student model
        # init_model=TrainModel.load_model(config_attack["path_student"])['model']
        # print(config_attack["path_student"])
        # print("student is the original non_watermarked model ")
        # student=deepcopy(init_model)

        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"}:
            _, bernon = self.watermark.extract(student, supp_wat_model,config_embed)

        else:
            _, bernon = self.watermark.extract(student, supp_wat_model)


        print("BER from the student model (non attacked) == ", bernon)

        student_state_dict = train_student(student, teacher, train_loader,
                                           epochs=config_attack["epoch_attack"], supp=supp_wat_model, device="cuda",
                      extract=self.watermark.extract, layer_name=config_attack["layer_name"], method_name=config_attack["method_wat"])#,  supp_attack=supp_ambiguity_wat_model)
        student.load_state_dict(student_state_dict)
        print("Check the accuracy of the student model")
        acc=TrainModel.evaluate(student, test_loader, config_embed)
        print(f"accuracy of the student model: {acc:.2f}%")
        if config_embed['method_wat'] in {"greedy", "Laplacian_dis"}:
            _, ber = self.watermark.extract(student, supp_wat_model,config_embed)
        else:
            _, ber = self.watermark.extract(student, supp_wat_model)

        print("BER of the Student after distillation == ", ber)


        save_dir = os.path.dirname(config_attack['save_path'])
        print(config_attack['save_path'])
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
        torch.save(student_state_dict, config_attack['save_path'])







    def distillation_hufu(self, config_embed, config_attack, config_data):
        # start by laoding the data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        train_loader_hufu, val_loader_hufu, test_loader_hufu = Hufu_func.create_data_loaders_hufu(
            batch_size=config_embed['batch_size_hufu'],
            validation_split=config_embed['validation_split']
        )


        # I need to load the watermarked model for having the watermarking parameters
        dict_model = TrainModel.load_model(config_embed['save_path'])
        dict_hufu=TrainModel.load_model(config_embed['save_path_hufu_finetune'])


        model_wat = dict_model["supplementary"]["model"].to(config_embed["device"])
        hufu_model=dict_hufu["supplementary"]["model"].to(config_embed["device"])


        print("Check the accuracy of the watermarked model")
        acc=TrainModel.evaluate(model_wat, test_loader, config_embed)
        print("acc wat model", acc)


        _, mse_after, _=self.watermark.extract(deepcopy(model_wat), deepcopy(hufu_model), dict_model["supplementary"]["selected_indexes"], train_loader_hufu, config_embed)
        print(f"watermark model:   mse HUFU before Distillation: {mse_after}")

        # I need to load the teacher model that just has been attacked with the dummy neurons and called the teacher
        # teacher = torch.load(config_attack['path_model'], weights_only=False).to(config_embed["device"])
        teacher =deepcopy(model_wat).to(config_embed["device"])

        #

        _, mse_after_t, _=self.watermark.extract(deepcopy(teacher), deepcopy(hufu_model), dict_model["supplementary"]["selected_indexes"], train_loader_hufu,
                     config_embed)
        print(f"teacher model:     mse HUFU before distillation: {mse_after_t}")


        # I need to load the student model not watermarked and check if it s not watermarked
        # student = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        student=torch.load(config_embed['path_model'], weights_only=False)['model']
        _, mse_after_s, _ = self.watermark.extract(deepcopy(student), deepcopy(hufu_model),
                                                                   dict_model["supplementary"]["selected_indexes"],
                                                                   train_loader_hufu,
                                                                   config_embed)
        print(f"mse HUFU by student model:   after student weight: {mse_after_s}")

        student = train_student_hufu(student, teacher, train_loader, temperature=2.0, lr=1e-3,
                                         epochs=config_attack["epoch_attack"], supp=dict_model["supplementary"],
                                         device="cuda",
                                         extract=self.watermark.extract, layer_name=config_attack["layer_name"],
                                     hufu_model=hufu_model,selected_indexes=dict_model["supplementary"]["selected_indexes"],
                                     train_loader_hufu=train_loader_hufu, config=config_embed)


        savedir = os.path.dirname(config_attack['save_path'])
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)


        _, mse_after, _=self.watermark.extract(deepcopy(student), deepcopy(hufu_model), dict_model["supplementary"]["selected_indexes"], train_loader_hufu, config_embed)


        acc = TrainModel.evaluate(deepcopy(student), test_loader, config_embed)
        print("acc student", acc)
        print(f"student model:    mse_by_teacher: {mse_after_t} mse_before distill: {mse_after_s}, mse_after distill: {mse_after}")




        torch.save(student, config_attack['save_path'])

    def fine_tune_attack_hufu(self, config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        train_loader_hufu, val_loader_hufu, test_loader_hufu = Hufu_func.create_data_loaders_hufu(
            batch_size=config_embed['batch_size_hufu'],
            validation_split=config_embed['validation_split']
        )

        # Get model
        # init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        # init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])


        # init_model.eval()
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        dict_hufu = TrainModel.load_model(config_embed['save_path_hufu_finetune'])

        model_wat_ = dict_model["supplementary"]["model"].to(config_embed["device"])
        hufu_model = dict_hufu["supplementary"]["model"].to(config_embed["device"])

        # fine tune the watermarked model
        print("Check the accuracy of the watermarked model")
        acc=TrainModel.evaluate(model_wat_, test_loader, config_attack)
        print("Compute the MSE from the watermarked model")
        _, mse_after, _ = self.watermark.extract(deepcopy(model_wat_), deepcopy(hufu_model), dict_model["supplementary"]["selected_indexes"],
                     train_loader_hufu, config_embed)
        print("MSE = ", mse_after)
        print("ACC = ", acc)
        print("Start fine-tuning")
        results_acc = []
        results_mse = []
        # print(config_attack["epochs"])
        # epochs = [config_attack["epochs"]*i for i in range(1, 4)]
        epochs = [10,10,10]

        # print(model_wat)
        # print("conf attack")
        # print(config_attack.keys())
        # print("conf embe")
        # print(config_embed.keys())
        # print("conf data")
        # print(config_data.keys())
        model_wat=deepcopy(model_wat_)
        x=1
        for ep in epochs:
            print("Epoch ", x* ep)
            config_attack["epochs"] = ep
            model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
            # check the accuracy and BER
            _, mse_after, _ = self.watermark.extract(deepcopy(model_wat), deepcopy(hufu_model),
                                           dict_model["supplementary"]["selected_indexes"],
                                           train_loader_hufu, config_embed)
            acc = TrainModel.evaluate(model_wat, test_loader, config_attack)
            print(f"ACC and MSE after finetuning {x*ep}")
            print("MSE = ", mse_after, "ACC = ", acc)
            results_acc.append(acc)
            results_mse.append(mse_after)
            x=x+1
        print("epochs = ", epochs)
        print("results_acc = ", results_acc)
        print("results_mse = ", results_mse)

    def pruning_attack_hufu(self, config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        train_loader_hufu, val_loader_hufu, test_loader_hufu = Hufu_func.create_data_loaders_hufu(
            batch_size=config_embed['batch_size_hufu'],
            validation_split=config_embed['validation_split']
        )

        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        dict_hufu = TrainModel.load_model(config_embed['save_path_hufu_finetune'])

        model_wat = dict_model["supplementary"]["model"].to(config_embed["device"])
        hufu_model = dict_hufu["supplementary"]["model"].to(config_embed["device"])

        # fine tune the watermarked model
        print("First check before pruning")
        acc=TrainModel.evaluate(model_wat, test_loader, config_attack)
        mse_before, mse_after, mse_non_wm=self.watermark.extract(deepcopy(model_wat), deepcopy(hufu_model), dict_model["supplementary"]["selected_indexes"],
                     train_loader_hufu, config_embed)
        print(f"acc: {acc}, mse_original: {mse_before}, mse_after_pruning: {mse_after}")
        results_acc = []
        results_mse = []
        pruning_rate = [x / 10 for x in range(10)] + [0.95, 0.99, 1.]

        for rate in pruning_rate:
            model = pruning(model_wat, rate)
            print("evaluate the model after pruning of amount", rate)
            results_acc.append(TrainModel.evaluate(model, test_loader, config_attack))
            mse_before, mse_after, mse_non_wm = self.watermark.extract(deepcopy(model), deepcopy(hufu_model),
                                                             dict_model["supplementary"]["selected_indexes"],
                                                             train_loader_hufu, config_embed)
            results_mse.append(mse_after)

        print("pruning_rate = ", pruning_rate)
        print("mse_original = ", mse_before)
        print("results_acc = ", results_acc)
        print("results_mse = ", results_mse)

    def train_embedding(self, config_data, config_embed):
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])

        # Get the training and testing data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)

        # embed the watermark
        model_wat, _= self.watermark.embed(init_model, test_loader, train_loader, config_embed)
        print(f"evaluate the watermarked model with {self.method}")
        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
        return acc

    def ambiguity_attack(self, config_embed, config_attack, config_data):
        # Get model

        if config_data["database"] == "cifar100":
            init_model_fake = deepcopy(torch.load(config_attack["path_model_fake"], weights_only=False)['model'])
            print("Initial fake model loaded from", config_attack["path_model_fake"])

            init_model_real = deepcopy(torch.load(config_attack["path_model"], weights_only=False)['model'])
            print("Initial real model loaded from", config_attack["path_model"])

            # Get data
            print("Load data loaders", config_data["database"])  # cifar100
            train_loader_100, test_loader_100 = Database.load_dataset_loaders(config_data)

        else:
            base_path = cf_cnn_dict["save_path"]
            print(base_path)
            path_real = base_path  # Assume this is the 100 epoch path
            path_fake = base_path.replace("_ep100_", "_ep50_")
            init_model_real = deepcopy(torch.load(  path_real, weights_only=False)['model'])
            print("Initial real model loaded from", path_real)
            init_model_fake = deepcopy(torch.load(path_fake, weights_only=False)['model'])
            print("Initial fake model loaded from", path_fake)
            init_model_real.eval()






        train_loader_10, test_loader_10 = Database.load_dataset_loaders(cf_cifar10_data) #cifar10
        # load the watermarked model

        dict_model = torch.load(config_embed['save_path'], weights_only=False)
        print('load watermarked model from', config_embed['save_path'])
        model_wat= deepcopy(dict_model["wat_model"].to(config_embed["device"]))


        # evaluate the models
        print("Evaluate the model before ambiguity attack")
        if config_data["database"] == "cifar100":
            acc_fake=TrainModel.evaluate(deepcopy(init_model_fake), test_loader_100, config_embed)
        else:

            acc_fake=TrainModel.evaluate(deepcopy(init_model_fake), test_loader_10, config_embed)

        acc_10 = TrainModel.evaluate(deepcopy(model_wat), test_loader_10, config_embed)
        acc_init_real=TrainModel.evaluate(deepcopy(init_model_real), test_loader_10, config_embed)

        print("ACC_fake_orig = ", acc_fake, "ACC_wat_model = ", acc_10, "ACC_real_orig = ", acc_init_real)

        BEr=self.watermark.extract(model_wat, dict_model)[1]
        print("BER before ambiguity attack", BEr)



        # embed the watermark with the ambiguity attack
        if config_data["database"] == "cifar100":
            origin_model_ambiguity,dict_model_attacked = ambiguity_attack(init_model_fake, model_wat,train_loader_100, test_loader_100,  config_attack)
        else:
            origin_model_ambiguity,dict_model_attacked = ambiguity_attack(init_model_fake, model_wat,train_loader_10, test_loader_10,  config_attack)
        # evaluate the attacked model
        print("Evaluate the model after ambiguity attack")
        if config_data["database"] == "cifar100":
            acc_fake_before_attack = TrainModel.evaluate(deepcopy(init_model_fake), test_loader_100, config_embed)
            acc_fake_after_attack = TrainModel.evaluate(deepcopy(origin_model_ambiguity), test_loader_100, config_embed)
        else:
            acc_fake_before_attack = TrainModel.evaluate(deepcopy(init_model_fake), test_loader_10, config_embed)
            acc_fake_after_attack = TrainModel.evaluate(deepcopy(origin_model_ambiguity), test_loader_10, config_embed)

        acc_10 = TrainModel.evaluate(deepcopy(model_wat), test_loader_10, config_embed)
        acc_init_real = TrainModel.evaluate(deepcopy(init_model_real), test_loader_10, config_embed)

        print("ACC_fake  before attack = ", acc_fake_before_attack, "ACC_ fake after attack = ", acc_fake_after_attack,"ACC_wat model = ", acc_10, "acc_real_orig = ", acc_init_real)

        b_ext_1, ber_1 = self.watermark.extract(origin_model_ambiguity, dict_model_attacked)
        b_ext_2, ber_2 = self.watermark.extract(origin_model_ambiguity, dict_model)
        b_ext_3, ber_3 = self.watermark.extract(init_model_fake, dict_model_attacked)
        b_ext_4, ber_4 = self.watermark.extract(init_model_fake, dict_model)
        b_ext_7, ber_7 = self.watermark.extract(init_model_real, dict_model_attacked)
        b_ext_8, ber_8 = self.watermark.extract(init_model_real, dict_model)

        b_ext_5, ber_5 = self.watermark.extract(model_wat, dict_model_attacked)
        b_ext_6, ber_6 = self.watermark.extract(model_wat, dict_model)

        print(f"BER_1 (fake model with fake projection model): {ber_1}")
        print(f"BER_2 (fake model with real watermark projection model): {ber_2}")
        print(f"BER_3 (fake original model with fake projection model): {ber_3}")
        print(f"BER_4 (fake original model with real watermark projection model): {ber_4}")
        print(f"BER_7 (real original model with fake projection model): {ber_7}")
        print(f"BER_8 (real original model with real watermark projection model): {ber_8}")



        print(f"BER_5 (watermarked model with fake projection model): {ber_5}")
        print(f"BER_6 (watermarked model with real watermark projection model): {ber_6}")

        return acc_fake_after_attack


    @staticmethod
    def pia_attack(config_data, config_embed, config_attack):
        # Train PIA Detector on 500 watermarked and 500 not watermarked models
        # And 200 models for testing



        # ************** Train the detector **************
        # load models and prepare the loaders
        nb_param = 100000

        results = torch.load(config_attack['save_path'])

        weights_ft = torch.cat([results["model_ft"][i].fc2.weight.data.flatten(0)[:nb_param].unsqueeze(0) for i in
                                results["model_ft"].keys()])
        labels_ft = torch.ones(size=(len(results["model_ft"].keys()), 1))

        weights_wat = torch.cat([results["model_ft_wat"][i].fc2.weight.data.flatten(0)[:nb_param].unsqueeze(0)
                                 for i in results["model_ft_wat"].keys()])
        labels_wat = torch.zeros(size=(len(results["model_ft_wat"].keys()), 1))

        (train_data_ft, test_data_ft) = torch.split(weights_ft, [700, 100])
        (train_labels_ft, test_labels_ft) = torch.split(labels_ft, [700, 100])

        (train_data_wat, test_data_wat) = torch.split(weights_wat, [700, 100])
        (train_labels_wat, test_labels_wat) = torch.split(labels_wat, [700, 100])

        train_data = torch.cat((train_data_ft, train_data_wat))
        train_labels = torch.cat((train_labels_ft, train_labels_wat))

        test_data = torch.cat((test_data_ft, test_data_wat))
        test_labels = torch.cat((test_labels_ft, test_labels_wat))

        print(train_data.shape, train_labels.shape)
        print(test_data.shape, test_labels.shape)
        #
        transform_train = transforms.Compose([
            # transforms.RandomCrop(size=32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # AddGaussianNoise(config["mean"], config["std"]),
        ])
        dataset_key = CustomTensorDataset(train_data, train_labels, transform_list=transform_train)
        key_loader_train = DataLoader(dataset=dataset_key, batch_size=64, shuffle=True)

        dataset_key = CustomTensorDataset(test_data, test_labels, transform_list=transform_train)
        key_loader_test = DataLoader(dataset=dataset_key, batch_size=200, shuffle=True)

        model_detector = PiaDetector(nb_param).to(config_data["device"])

        optimizer = optim.Adam(model_detector.parameters(), lr=1e-1)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        epochs = 100

        for epoch in range(epochs):
            train_loss = 0
            loop = tqdm(key_loader_train, leave=True)
            epoch_acc = 0
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(config_data["device"]), targets.to(config_data["device"])
                # targets = targets.type(torch.long)
                optimizer.zero_grad()
                outputs = model_detector(inputs)
                loss = criterion(outputs, targets)
                # back propagate the loss
                loss.backward()
                optimizer.step()
                acc = Metric.binary_acc(outputs, targets)

                train_loss += loss.item()
                epoch_acc += acc.item()

                # update the progress bar
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=f"{epoch_acc / (batch_idx + 1):.3f}")

    @staticmethod
    def gen_database(config_data):
        """ generate a new database"""
        Database.gen_dataset_loaders(config_data)

    @staticmethod
    def train_model(config_data, config_train):
        # get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        print("training sample size:",len(train_loader.dataset))
        print("test sample size:", len(test_loader.dataset))
        # get model
        if config_data['n_classes'] :
            init_model = TrainModel.get_model(config_train["architecture"], config_train["device"],config_data["n_classes"])
        else:
            init_model = TrainModel.get_model(config_train["architecture"], config_train["device"])
        print("Model to train...")
        print(init_model)
        """Start training the model"""
        if config_train["show_acc_epoch"]:
            _, acc_list = TrainModel.fine_tune(init_model, train_loader, test_loader, config_train)
            print(acc_list[-1])
            Tests.plot_acc(acc_list, config_train)
        else:
            TrainModel.fine_tune(init_model, train_loader, test_loader, config_train)

    @staticmethod
    def plot_acc(acc_list, config_train):
        epochs = np.arange(len(acc_list))
        plt.plot(epochs, acc_list, c="black", marker='*', label=f"ACC of {config_train['architecture']} "
                                                                f"over database = {config_train['database']}")
        # for a, acc in zip(epochs, acc_list):
        #     plt.text(a + 0.02, acc + 0.02, str(acc))
        plt.xticks(np.arange(-1, len(acc_list) + 2, 1))
        plt.yticks(np.arange(-10, 110, 10))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.grid(True)
        plt.title(f"ACC of {config_train['architecture']} over database = {config_train['database']}")
        plt.savefig(config_train['save_fig_path'])
        plt.show()

    @staticmethod
    def get_watermark(method):
        match method:
            case "DICTION":
                from watermark.diction import Diction
                return Diction()
            case "DICTION2":
                from watermark.diction2 import Diction2
                return Diction2()
            case "DEEPSIGNS":
                from watermark.deepsigns import Deep_Signs
                return Deep_Signs()
            case "UCHIDA":
                from watermark.uchida import Uchida
                return Uchida()
            case "RES_ENCRYPT":
                from watermark.res_encrypt import Res_Encrypt
                return Res_Encrypt()
            case "RIGA":
                from watermark.riga import Riga
                return Riga()
            case "STDM":
                from watermark.stdm import Stdm
                return Stdm()
            case "PASSPORT":
                from watermark.passport import Passport
                return Passport()
            case "HUFUNET":
                from watermark.hufunet import HUFUNET
                return HUFUNET()
            case "GREEDY":
                from watermark.greedy import Greedy
                return Greedy()
            case "LAPLACIAN":
                from watermark.laplacian_dis import Laplacian
                return Laplacian()
            case _:
                raise ValueError(f"method {method} not implemented")