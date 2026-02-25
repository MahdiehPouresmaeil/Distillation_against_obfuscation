from copy import deepcopy

import torch
from torch import optim, nn
from torch.nn import BCELoss
from tqdm import tqdm

from networks.linear_mod import EncResistant
from util.metric import Metric
from util.util import Random, TrainModel


from watermark.watermark  import Watermark

class Res_Encrypt(Watermark):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name():
        return "Res_Encrypt"

    def keygen(self, watermark_size):
        """
        Generate a secret key and message.
        """
        # Generate the watermark
        watermark = torch.tensor(Random.get_rand_bits(watermark_size, 0., 1.)).cuda()
        # Here the secret matrix is generated in the mapping net
        return watermark

    def embed(self, init_model, test_loader, train_loader, config) -> object:
        # Generate the watermark to embed
        watermark = self.keygen(config["watermark_size"])
        # Generate a random watermark
        watermark_rd = self.keygen(config["watermark_size"])
        # Instance the target and the MappingNet model and move them to the device
        model = deepcopy(init_model)
        init_model = deepcopy(init_model)
        # make sure that the original will not be modified by the watermarking
        for param in init_model.parameters():
            param.requires_grad = False
        # init_model.eval()

        layer_params = [param for name, param in model.named_parameters() if name == config["layer_name"]]
        # Ensure we found the layer and it has parameters
        if not layer_params:
            raise ValueError("No layer found with the specified name")
        weights_selected_layer = layer_params[0]

        # Checking the shape of the selected weights
        print("Selected layer shape:", weights_selected_layer.shape)
        theta_f = torch.flatten(self._get_mean(weights_selected_layer))
        print("theta_f shape", theta_f.shape)
        weight_size = len(theta_f)
        # instantiation of the Mapping net
        mapping_net = EncResistant(config, weight_size).cuda()

        # Get the weights of the original model
        weights_selected_layer_init = [param for name, param in init_model.named_parameters() if
                                       name == config["layer_name"]][0]

        theta_fn = torch.flatten(self._get_mean(weights_selected_layer_init))

        # Loss and optimizer
        criterion = config["criterion"]
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
            {'params': mapping_net.parameters(), 'lr': config["lr_MN"], 'weight_decay': config["wd_MN"]}
        ], lr=config["lr"])

        ber_ = l_global = l_wat_orig = l_l1_w = l_wat = 1
        scheduler = TrainModel.get_scheduler(optimizer, config)

        for epoch in range(config["epochs"]):
            train_loss = correct = total = 0
            loop = tqdm(train_loader, leave=True)
            model.train(True)
            mapping_net.train(True)
            for batch_idx, (x_train, y_train) in enumerate(loop):
                # Move tensors to the configured device
                x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
                optimizer.zero_grad()
                y_pred = model(x_train)
                # Compute the losses
                weights_selected_layer = [param for name, param in model.named_parameters() if
                                          name == config["layer_name"]][0]
                theta_f = torch.flatten(self._get_mean(weights_selected_layer))

                matrix_g = mapping_net(theta_f.cuda())



                matrix_gn = mapping_net(theta_fn.cuda())

                # sanity check
                assert BCELoss(reduction='sum')(matrix_g, watermark).requires_grad is True, 'broken computational graph :/'

                l_main_task = criterion(y_pred, y_train)
                l_wat = BCELoss(reduction='sum')(matrix_g, watermark)
                l_wat_orig = BCELoss(reduction='sum')(matrix_gn, watermark_rd)
                l_l1_w = torch.norm(theta_f, p=1)
                l_global = (l_main_task + config["lambda_1"] * l_wat + config["lambda_2"] * l_wat_orig +
                            config["lambda_3"] * l_l1_w)

                # Backpropagation and optimization
                l_global.backward(retain_graph=True)
                optimizer.step()

                train_loss += l_main_task.item()
                _, predicted = y_pred.max(1)
                total += y_train.size(0)
                correct += predicted.eq(y_train).sum().item()

                assert l_global.requires_grad is True, 'broken computational graph :/'

                _, ber_ = self._extract(model, watermark, config["layer_name"], mapping_net)

                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(l_main_task=train_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"
                                               f"/{total}]", l_wat=f"{l_wat:1.4f}", l_wat_orig=f"{l_wat_orig:1.4f}",
                                 loss_l1=f"{l_l1_w.item():1.4f}",
                                 ber_=f"{ber_:1.3f}")
            scheduler.step()
            if (epoch + 1) % config["epoch_check"] == 0:
                with torch.no_grad():
                    _, ber_ = self._extract(model, watermark, config["layer_name"], mapping_net)
                    acc = TrainModel.evaluate(model, test_loader, config)
                    print(
                        f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber_:1.3f}---l_wat: "
                        f"{l_wat:1.4f}---l_wat_orig: {l_wat_orig:.3f}---l_l1_w: {l_l1_w.item():.3f}---acc: {acc}")

                if ber_ == 0:
                    print("saving... watermarked model! ")
                    supplementary = {'model': model, 'watermark': watermark, 'ber': ber_,
                                     "layer_name": config["layer_name"], "mapping_net": mapping_net}

                    self.save(config["save_path"], supplementary)
                    # TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
                    print("model saved!")
                    break
        return model, ber_


    def extract(self, classifier, supp):
        return self._extract(classifier, supp["watermark"], supp["layer_name"],
                             supp["mapping_net"])


    def _extract(self,model, watermark, layer_name, mapping_net):
        weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name]
        weights_selected_layer = weights_selected_layer[0]
        theta_f = torch.flatten(self._get_mean(weights_selected_layer))
        theta_f = theta_f.unsqueeze(0)

        g_ext = mapping_net(theta_f)
        ber = self._get_ber(g_ext[0], watermark)
        return g_ext[0], ber

    def _get_mean(self, weights_selected_layer):
        # Determine the layer type based on dimensions of weights (generic way)
        if len(weights_selected_layer.shape) == 4:  # Usually, conv layers have 4D weights [out_channels, in_channels, h, w]
            # Mean across each kernel
            mean_by_kernel = weights_selected_layer.mean(dim=(0, 1))  # Averaging over in_channels, height, and width
            return mean_by_kernel
        elif len(weights_selected_layer.shape) == 2:  # Linear layers have 2D weights [out_features, in_features]
            # Mean across each perceptron
            mean_by_perceptron = weights_selected_layer.mean(dim=0)  # Averaging over input features
            return mean_by_perceptron
        else:
            raise NotImplementedError(
                "The layer type is not supported or cannot be determined based on the weight dimensions")
