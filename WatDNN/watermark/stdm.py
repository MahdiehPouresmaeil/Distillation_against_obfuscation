from copy import deepcopy

import torch.nn.functional as F
import torch
from sympy.abc import alpha
from torch import optim, nn
from torch.nn import BCELoss
from tqdm import tqdm


from util.util import Random, TrainModel

from watermark.watermark import Watermark

class Stdm(Watermark):
    def __init__(self):
        super().__init__()


    @staticmethod
    def get_name():
        return "STDM" # Short for "Spread Transform Dither Modulation Watermarking"

    def keygen(self, watermark_size, selected_weights_size):
        """
        Generate a secret key and message.
        """
        # Generate the watermark
        watermark = torch.tensor(Random.get_rand_bits(watermark_size, 0., 1.)).cuda()
        # Generate matrix matrix_a normalize l2 p=2 on col dim=0
        matrix_a = 1. * torch.randn(selected_weights_size, watermark_size, requires_grad=False).cuda()
        matrix_a = F.normalize(matrix_a, p=2, dim=0).cuda()
        return watermark, matrix_a


    @staticmethod
    def _theta(x, alpha_, beta_):
        numerator = torch.exp(alpha_ * torch.sin(torch.tensor(beta_) * x))
        denominator = 1 + torch.exp(alpha_ * torch.sin(torch.tensor(beta_) * x))
        return numerator / denominator


    def embed(self, init_model, test_loader, train_loader, config) -> object:

        # Instance the target model and  perceptron
        model = deepcopy(init_model)

        weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]][0]
        selected_weights = torch.flatten(weights_selected_layer.mean(dim=0))
        print("Selected weights shape ", selected_weights.shape)
        # Generate a random watermark to insert and a normalized matrix_a with l2 p=2 on col dim=0
        watermark, matrix_a = self.keygen(config["watermark_size"], len(selected_weights))

        # Loss and optimizer
        criterion = config["criterion"]
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = TrainModel.get_scheduler(optimizer, config)

        ber_ = l_wat = l_global = 1
        matrix_g = 0

        for epoch in range(config["epochs"]):
            train_loss = correct = total = 0
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (inputs, targets) in enumerate(loop):
                # Move tensors to the configured device
                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute the loss
                weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]][0]
                selected_weights = torch.flatten(weights_selected_layer.mean(dim=0))

                matrix_g = self._theta(selected_weights @ matrix_a, config["alpha"], config["beta"])

                # sanity check
                assert BCELoss(reduction='sum')(matrix_g, watermark).requires_grad is True, 'broken computational graph :/'
                # λ, control the trade of between WM embedding and Training
                l_main_task = criterion(outputs, targets)
                l_wat = BCELoss(reduction='sum')(matrix_g, watermark)
                l_global = l_main_task + config["lambda_1"] * l_wat

                train_loss += l_main_task.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                assert l_global.requires_grad is True, 'broken computational graph :/'
                # Backpropagation and optimization
                l_global.backward(retain_graph=True)
                optimizer.step()

                ber = self._get_ber(matrix_g, watermark)
                _, ber_ = self._extract(model, matrix_a, watermark, config["layer_name"], config["alpha"], config["beta"])

                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"
                                               f"/{total}]",  l_wat=f"{l_wat:1.4f}",
                                 ber=f"{ber:1.3f}",
                                 ber_=f"{ber_:1.3f}")
            scheduler.step()

            if (epoch + 1) % config["epoch_check"] == 0:
                with torch.no_grad():
                    ber = self._get_ber(matrix_g, watermark)
                    _, ber_ = self._extract(model, matrix_a, watermark, config["layer_name"], config["alpha"], config["beta"])
                    acc = TrainModel.evaluate(model, test_loader, config)
                    print(
                        f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}"
                        f"--l_wat: {l_wat:1.4f}---acc: {acc}")

                if ber_ == 0:
                    print("saving!")
                    supplementary = {'model': model, 'matrix_a': matrix_a, 'watermark': watermark, 'ber': ber,
                                     "layer_name": config["layer_name"], "alpha": config["alpha"], "beta": config["beta"]}

                    self.save(path=config['save_path'], supplementary=supplementary)
                    print("model saved!")
                    break

        return model, ber_


    def extract(self, classifier, supp):
        return self._extract(classifier, supp["matrix_a"], supp["watermark"], supp["layer_name"], supp["alpha"], supp["beta"])


    def _extract(self, model, matrix_a, watermark, layer_name, alpha_param, beta_param):
        weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name][0]
        w = torch.flatten(weights_selected_layer.mean(dim=0))
        g_ext = self._theta(w @ matrix_a,alpha_param, beta_param)
        ber = self._get_ber(g_ext, watermark)
        return g_ext, ber
