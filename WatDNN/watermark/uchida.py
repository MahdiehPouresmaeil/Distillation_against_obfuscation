from copy import deepcopy

import torch
from torch import optim, nn
from torch.nn import BCELoss
from tqdm import tqdm

from util.util import Random, TrainModel


from watermark.watermark  import Watermark

class Uchida(Watermark):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name():
        return "UCHIDA"

    def keygen(self, watermark_size, selected_weights_size):
        """
        Generate a secret key and message.
        """
        # Generate the watermark
        watermark = torch.tensor(Random.get_rand_bits(watermark_size, 0., 1.)).cuda()
        # Generate matrix matrix_a
        matrix_a = 1. * torch.randn(selected_weights_size, watermark_size, requires_grad=False).cuda()
        return watermark, matrix_a

    def embed(self, init_model, test_loader, train_loader, config) -> object:
        # Instance the target model and Uchida perceptron
        model = deepcopy(init_model)
        print("the names of the layers of the target model...")
        for name, param in model.named_parameters():
            print(name)
        # Select the weights
        print("watermark layer name ", config["layer_name"])
        weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]][0]
        selected_weights = torch.flatten(weights_selected_layer.mean(dim=0))
        # model_uchida = Uchida(selected_weights)
        print("Selected weights shape ", selected_weights.shape)
        # Generate a random watermark and matrix A
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

                matrix_g = torch.nn.Sigmoid()(selected_weights @ matrix_a)

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
                _, ber_ = self._extract(model, matrix_a, watermark, config["layer_name"])

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
                    _, ber_ = self._extract(model, matrix_a, watermark, config["layer_name"])
                    acc = TrainModel.evaluate(model, test_loader, config)
                    print(
                        f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}"
                        f"--l_wat: {l_wat:1.4f}---acc: {acc}")

                if ber_ == 0:
                    print("saving!")
                    supplementary = {'model': model, 'matrix_a': matrix_a, 'watermark': watermark, 'ber': ber,
                                     "layer_name": config["layer_name"]}

                    self.save(path=config['save_path'], supplementary=supplementary)
                    print("model saved!")
                    break

        return model, ber_

    def extract(self, model_watermarked, supp):
        return self._extract(model_watermarked, supp["matrix_a"], supp["watermark"], supp["layer_name"])

    def _extract(self, model, matrix_a, watermark, layer_name):
        weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name][0]

        w = torch.flatten(weights_selected_layer.mean(dim=0))
        # print("w",w)
        g=w @ matrix_a
        # print("g",g)
        g_ext = torch.nn.Sigmoid()(g)
        # print("extract", g_ext)
        ber = self._get_ber(g_ext, watermark)
        # print("ber", ber)
        return g_ext, ber

