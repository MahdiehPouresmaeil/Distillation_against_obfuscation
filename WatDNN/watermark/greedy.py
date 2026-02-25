from watermark.watermark import Watermark
import rsa
import torch
from torch import optim
import torch.nn.functional as F
from copy import deepcopy
from util.util import TrainModel
from torch.nn import BCELoss
from tqdm import tqdm
import sys


class Greedy(Watermark):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name():
        return "GREEDY"

    # def keygen(self, message, config):
    #     """
    #     Generate a secret key and message.
    #     """
    #     (pubkey, privkey) = rsa.newkeys(512)
    #     message = message.encode('utf-8')
    #     signature = rsa.sign(message, privkey, 'SHA-256')
    #     signature_set = rsa.compute_hash(signature, 'SHA-256')
    #     b_sig = list(bin(int(signature_set.hex(), 16))[2:].zfill(256))  # hex -> bin always 256 bit
    #     # print(b_sig)
    #     # print(len(b_sig))
    #     # b_sig = list(bin(int(signature_set.hex(), base=16)).lstrip('0b'))  # hex -> bin
    #     # print(b_sig)
    #     # print(len(b_sig))
    #
    #     b_sig = list(map(int, b_sig))  # bin-> int
    #     print(b_sig)
    #     sig = torch.tensor([-1 if i == 0 else i for i in b_sig], dtype=torch.float)
    #     watermark = torch.tensor(b_sig, dtype=torch.float32, device=config["device"])
    #     # print(sig.size())
    #
    #     return sig, watermark

    def keygen(self, message):
        """
        Génère un watermark aléatoire basé sur le message.
        Returns:
            watermark_sig: Watermark en {-1, +1}
            watermark: Watermark binaire en {0, 1}
        """
        watermark_size = self.config.get("watermark_size", 128)

        # Set seed based on message for reproducibility
        seed = hash(message) % (2 ** 32)
        torch.manual_seed(seed)

        # Generate random binary watermark {0, 1}
        b_sig = torch.randint(0, 2, (watermark_size,)).tolist()

        # Convert to signs: 0 -> -1, 1 -> +1
        watermark_sig = torch.tensor(
            [-1.0 if i == 0 else 1.0 for i in b_sig],
            dtype=torch.float
        ).to(self.device)

        # Binary watermark {0, 1}
        watermark = torch.tensor(b_sig, dtype=torch.float32).to(self.device)

        return watermark_sig, watermark

    def embed(self, init_model, test_loader, train_loader, config) -> object:

        watermark_sig, watermark = self.keygen(config["message"],
                                               config)  # the watermark message in {1,+1} with size 256

        print((watermark_sig))
        weight_extract = self.weight_extraction(deepcopy(init_model), config)
        print((weight_extract))
        residual_vector = self.residuals(weight_extract, config)
        print((residual_vector))

        print(watermark_sig.shape)
        print((residual_vector.shape))

        model = deepcopy(init_model)

        # Loss and optimizer
        criterion = config["criterion"]
        # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0.0005)
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = TrainModel.get_scheduler(optimizer, config)

        ber_ = l_wat = l_global = 1

        for epoch in range(config["epochs"]):
            # print("epoch:", epoch)
            train_loss = correct = total = 0
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (inputs, targets) in enumerate(loop):
                # print("Batch idx:", batch_idx)
                # print("epoch:", epoch)
                # Move tensors to the configured device

                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                optimizer.zero_grad()
                outputs = model(inputs)

                weight_extract = self.weight_extraction(model, config)

                residual_vector = self.residuals(weight_extract, config)
                # print((residual_vector))

                # Compute the loss
                # λ, control the trade of between WM embedding and Training
                l_main_task = criterion(outputs, targets)
                # print(f"l_main_task", l_main_task)

                l_wat = F.relu(
                    config['treshold'] - (watermark_sig.view(-1).cuda() * residual_vector.view(-1).cuda())).sum()
                # print(f"l_wat", l_wat)
                l_global = l_main_task + config["lambda_1"] * l_wat

                train_loss += l_main_task.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                l_global.backward()
                optimizer.step()

                _, ber = self._extract(model, watermark, config)

                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"
                                               f"/{total}]", l_wat=f"{l_wat:1.4f}",
                                 ber=f"{ber:1.3f}"
                                 )
            scheduler.step()

            if (epoch + 1) % config["epoch_check"] == 0:
                with torch.no_grad():

                    _, ber = self._extract(model, watermark, config)
                    acc = TrainModel.evaluate(model, test_loader, config)
                    print(
                        f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber:1.3f}"
                        f"--l_wat: {l_wat:1.4f}---acc: {acc}")

                if ber == 0:
                    print("saving!")
                    supplementary = {'model': model, 'watermark': watermark, 'ber': ber,
                                     "layer_name": config["layer_name"]}

                    self.save(path=config['save_path'], supplementary=supplementary)
                    print("model saved!")
                    break

        return model, ber

    def extract(self, model_watermarked, supp, config):
        return self._extract(model_watermarked, supp["watermark"], config)

    def _extract(self, model, watermark, config):

        weight_extract = self.weight_extraction(deepcopy(model), config)
        residual_vector = self.residuals(weight_extract, config)
        signs = torch.sign(residual_vector)
        signs[signs == -1] = 0


        ber = self._get_ber(signs, watermark)
        ber2 = float(ber)

        return signs, ber2

    def weight_extraction(self, model, config):

        for i, (name, param) in enumerate(model.named_parameters()):
            if name == config["layer_name"]:
                # print(name, param.shape)
                layer = param.view(-1)
                # print(layer)
                layer_weight = param.view(-1)[
                    :param.numel() // config['objective_size'][1] * config['objective_size'][1]]

                # print(layer_weight)
                layer_weight = F.adaptive_avg_pool1d(layer_weight[None, None],
                                                     config['objective_size'][0] * config['objective_size'][1]).squeeze(
                    0).view(config['objective_size'])
                # print(layer_weight)

        return layer_weight  # return the gama weights

    def residuals(self, weight_extraction, config):
        num_rows, num_cols = weight_extraction.shape
        # print((num_rows, num_cols))

        # Number of weights to prune (smallest ones) per row
        num_to_prune = int(num_cols * config['ratio'] + 0.5)
        # print("num_to_prune", num_to_prune)

        # Sort weights in each row by absolute value
        sorted_indices = torch.argsort(torch.abs(weight_extraction), dim=1)

        # Take the indices of the "num_to_prune" smallest weights in each row
        prune_indices = sorted_indices[:, :num_to_prune]
        # print("prune_indices", prune_indices.shape)

        # Zero out those smallest weights row by row
        for row in range(num_rows):
            weight_extraction[row, prune_indices[row]] = 0

        # After pruning, compute the mean of each row

        residual_vector = torch.mean(weight_extraction, dim=1)

        return residual_vector

