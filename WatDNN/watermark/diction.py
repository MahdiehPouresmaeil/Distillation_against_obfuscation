import sys
from copy import deepcopy
import random
import torch
from torch import nn, optim
from torch.nn import BCELoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm

from networks.linear_mod import ProjMod
from util.util import TrainModel, Random, CustomTensorDataset, AddGaussianNoise, AddWhiteSquareTransform, Util, AddWatermarkPatternTransform
from watermark.watermark import Watermark


def input_extractor_concat(extractor, x, layer_names, device="cuda"):
    """
    Extrait les activations de plusieurs couches, les redimensionne
    à la plus petite résolution spatiale (H, W) trouvée, et les concatène.

    Returns:
        concat_tensor: Tensor [Batch, Total_Channels, Min_H, Min_W]
        total_channels: int (Somme des canaux concaténés)
    """
    # 1. Extraction brute
    features = extractor(x.to(device)) #It returns a dictionary where keys are layer names and values are the corresponding feature tensors.
    outputs = [features[name] for name in layer_names]

    # 2. Trouver la taille spatiale minimale (H, W)
    # Chaque output est supposé être [Batch, Channel, H, W]
    min_h = min([t.shape[2] for t in outputs])
    min_w = min([t.shape[3] for t in outputs])
    target_size = (min_h, min_w)

    # 3. Redimensionner toutes les couches vers cette taille cible
    resized_outputs = []
    total_channels = 0

    for t in outputs:
        current_channels = t.shape[1]
        total_channels += current_channels

        # Si la taille diffère, on interpole
        if t.shape[2:] != target_size:
            # Interpolation bilinéaire pour préserver l'info spatiale
            #make the bigger feature as the size of the smaller
            t_resized = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
            resized_outputs.append(t_resized)
        else:
            resized_outputs.append(t)

    # 4. Concaténer sur la dimension des channels (dim=1)
    concat_tensor = torch.cat(resized_outputs, dim=1)
    # print(concat_tensor.shape)
    # concat_tensor=concat_tensor.sum(dim=1).unsqueeze(1)
    concat_tensor = concat_tensor.mean(dim=1).unsqueeze(1)
    # print(concat_tensor.shape)
    return concat_tensor, total_channels


class Diction(Watermark):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name():
        return "DICTION"

    def keygen(self, watermark_size):
        """
        Generate a secret key and message.
        """
        watermark = Random.get_rand_bits(watermark_size, 0., 1.)
        watermark = torch.tensor(watermark).reshape(1, watermark_size)
        return watermark

    def embed(self, init_model, test_loader, train_loader, config) -> object:

        # 1. Préparation des clés et modèles
        watermark = self.keygen(config["watermark_size"])
        watermark_rd = self.keygen(config["watermark_size"])
        # watermark_rd = torch.full((1, config["watermark_size"]), 0.5)

        wat_model = deepcopy(init_model)
        orig_model_copy = deepcopy(init_model)

        # Création des extracteurs pour récupérer les couches intermédiaires
        extractor_wat = create_feature_extractor(wat_model, config["layer_name"])
        extractor_orig = create_feature_extractor(orig_model_copy, config["layer_name"])

        # Affichage du graph pour debug
        # print("Model graph nodes:", get_graph_node_names(orig_model_copy)[0])

        # 2. Préparation du Trigger Set (Backdoor Strategy)
        # On récupère un batch d'images réelles pour base
        x_key_base, _ = next(iter(train_loader))

        # --- MODIFICATION: On force les labels à 0 (Backdoor) ---
        y_key = torch.zeros(len(x_key_base), dtype=torch.long)

        # Application des transformations (Bruit, Carré blanc, etc.)
        transform_train = transforms.Compose([
            # AddGaussianNoise(config["mean"], config["std"]),
            # AddWhiteSquareTransform(square_size=config["square_size"], start_x=config["start_x"],
            #                         start_y=config["start_y"]),
            AddWatermarkPatternTransform(
                 pattern_size=10,
                 start_x=0,
                 start_y=0,
                 intensity=1.0,
                 pattern_type='high_freq')
        ])

        dataset_key = CustomTensorDataset(x_key_base, y_key, transform_list=transform_train)
        key_loader = DataLoader(dataset=dataset_key, batch_size=config["batch_size"], shuffle=False)

        # 3. Calcul dynamique des dimensions pour ProjMod (Concaténation)
        print("Calcul des dimensions pour la concaténation...")
        x_key_sample, _ = next(iter(key_loader))

        with torch.no_grad():
            # On passe un sample pour voir quelle taille sortira de la concaténation
            _, total_channels = input_extractor_concat(
                extractor_orig, x_key_sample, config["layer_name"], device=config["device"]
            )

        print(f"Total channels after concatenation: {total_channels}")

        # Mise à jour de la config pour ProjMod
        config["total_channels"] = 1 #total_channels

        # Instanciation du modèle de projection
        # Assurez-vous que ProjMod dans linear_mod.py accepte 'total_channels' et traite du 4D
        proj_mod = ProjMod(config).to(config["device"])

        # 4. Configuration de l'entraînement
        criterion = config["criterion"]

        optimizer_proj = optim.AdamW(proj_mod.parameters(), lr=config["lr_proj"], weight_decay=config["wd_proj"])
        optimizer_model = optim.AdamW(wat_model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        optimizer_model_orig = optim.AdamW(orig_model_copy.parameters(), lr=config["lr"], weight_decay=config["wd"])

        proj_scheduler = TrainModel.get_scheduler(optimizer_proj, config)
        model_scheduler = TrainModel.get_scheduler(optimizer_model, config)

        ber_ = ber = l_proj = 1

        # 5. Boucle d'entraînement
        for epoch in range(config["epochs"]):
            embed_loss = train_loss = correct = train_loss_orig = correct_orig = total = 0
            loop = tqdm(train_loader, leave=True)

            proj_mod.train(True)
            wat_model.train(True)
            extractor_wat.train(True)

            for batch_idx, (x_train, y_train) in enumerate(loop):

                # Récupération du Trigger Set
                x_key, y_key = next(iter(key_loader))
                x_key = x_key.to(config["device"])
                y_key = y_key.to(config["device"])

                # --- A. Entraînement de ProjMod (Discriminateur) --- ###################################################
                optimizer_proj.zero_grad()

                # Extraction & Concaténation (Modèle Tatoué)
                act_wat, _ = input_extractor_concat(extractor_wat, x_key, config["layer_name"], config["device"])

                # Extraction & Concaténation (Modèle Original)
                act_orig, _ = input_extractor_concat(extractor_orig, x_key, config["layer_name"], config["device"])

                # Projection
                #by detach act will be treated as constant, no gradient will be computed for it, gradiant will flow into proj_mod only
                # print(act_wat.shape)

                watermark_out = proj_mod(act_wat.detach())
                watermark_orig_out = proj_mod(act_orig.detach())

                # Loss Proj
                l_wat = nn.BCELoss(reduction='mean')(watermark_out, watermark.repeat(len(watermark_out), 1).cuda())
                l_wat_orig = nn.BCELoss(reduction='mean')(watermark_orig_out,
                                                          watermark_rd.repeat(len(watermark_out), 1).cuda())

                l_proj = l_wat_orig + l_wat

                l_proj.backward(retain_graph=True)
                optimizer_proj.step()

                # --- B. Entraînement du Modèle Tatoué (Générateur) ---###################################################
                optimizer_model.zero_grad()
                x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
                y_train = y_train.type(torch.long)

                # Tâche Principale (Classification)
                y_pred = wat_model(x_train)
                l_main_task = criterion(y_pred, y_train)

                # Tâche Trigger Set (Backdoor: forcer la prédiction 0 sur le bruit)
                y_pred_key = wat_model(x_key)
                l_main_task_key = criterion(y_pred_key, y_key)

                # Tâche Tatouage (Modification des activations internes)

                # Extraction & Concaténation (Modèle Original)
                act_orig, _ = input_extractor_concat(extractor_orig, x_key, config["layer_name"], config["device"])

                # On ré-extrait car les poids de wat_model changent (besoin des gradients ici)
                act_wat, _ = input_extractor_concat(extractor_wat, x_key, config["layer_name"], config["device"])
                watermark_out = proj_mod(act_wat)
                l_wat1 = nn.BCELoss(reduction='mean')(watermark_out, watermark.repeat(len(watermark_out), 1).cuda())
                diff_loss = F.cosine_similarity(
                                act_wat.flatten(1),
                                act_orig.flatten(1),
                                dim=1
                            ).mean().item()# nn.MSELoss(reduction='mean')(act_wat, act_orig.detach())

                # Loss Totale
                l_model = 2*l_main_task + 1 * l_wat1 +1* l_main_task_key #- 0.01*diff_loss

                l_model.backward()
                optimizer_model.step()

                # --- C. Entraînement Modèle Original (Freeze ou Update léger si nécessaire) --- ###################################################
                # Note: Souvent on freeze l'original, mais votre code original l'entraînait aussi
                optimizer_model_orig.zero_grad()
                y_pred_orig = orig_model_copy(x_train)
                l_main_task_orig = criterion(y_pred_orig, y_train)
                l_main_task_orig.backward()
                optimizer_model_orig.step()

                # --- Logging ---
                train_loss += l_main_task.item()
                train_loss_orig += l_main_task_orig.item()
                embed_loss += l_proj.item()

                _, predicted = y_pred.max(1)
                _, predicted_orig = y_pred_orig.max(1)
                total += y_train.size(0)
                correct += predicted.eq(y_train).sum().item()
                correct_orig += predicted_orig.eq(y_train).sum().item()

                ber = self._get_ber(watermark_out, watermark.repeat(len(watermark_out), 1))

                grad_proj = log_gradients(proj_mod, "Proj")
                grad_wat = log_gradients(wat_model, "WatModel")

                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(
                    acc=f"{100. * correct / total:.1f}",
                    l_proj=f"{embed_loss / (batch_idx + 1):1.4f}",
                    ber=f"{ber:1.3f}",
                    l_wat=f"{l_wat:1.4f}",
                    l_wat_orig=f"{l_wat_orig:1.4f}",
                    L_wat1=f"{l_wat1:1.4f}",

                    Grad=f"Proj: {grad_proj:.4f} | WatModel: {grad_wat:.4f}"
                )

            model_scheduler.step()
            proj_scheduler.step()

            # --- Checkpoint & Save ---
            if (epoch + 1) % config["epoch_check"] == 0:
                with torch.no_grad():
                    # Test sur le Trigger Set
                    act_wat, _ = input_extractor_concat(extractor_wat, x_key, config["layer_name"], config["device"])
                    _, ber_ = self._extract(act_wat, proj_mod, watermark.cuda())

                    # Test Précision sur Test Set
                    acc = TrainModel.evaluate(wat_model, test_loader, config)
                    print(f"epoch:{epoch}---l_proj: {l_proj.item():1.5f}---ber_: {ber_:1.3f}---acc: {acc}")

                if ber_ == 0:
                    print("Saving watermarked model...")
                    supplementary = {
                        'wat_model': wat_model,
                        'matrix_a': proj_mod,
                        'watermark': watermark,
                        'watermark_r': watermark_rd,
                        'x_key': x_key,
                        'y_key': y_key,
                        'key_loader': key_loader,
                        'ber': ber,
                        "layer_name": config["layer_name"],
                        "total_channels": total_channels  # Important pour charger ProjMod plus tard
                    }
                    self.save(path=config['save_path'], supplementary=supplementary)
                    print("Model saved!")
                    break

        return wat_model, ber_

    def extract(self, classifier, supp):
        # Configuration
        # classifier.train(True)

        extractor = create_feature_extractor(deepcopy(classifier), supp["layer_name"])
        extractor.train()  # Important pour BatchNorm/Dropout si utilisés

        x_key = supp["x_key"]


        # Extraction avec la méthode concaténée
        act, _ = input_extractor_concat(extractor, x_key, supp["layer_name"], device="cuda")

        # Appel interne
        wat_ext, ber = self._extract(act.cuda(), supp["matrix_a"], supp["watermark"])
        return wat_ext, ber

    def _extract(self, act, model, watermark):




        watermark_out = model(act)


        # Moyenne sur le batch pour obtenir un vecteur unique
        watermark_out = torch.mean(watermark_out, dim=0, keepdim=True)
        #print(watermark_out)
        ber = self._get_ber(watermark_out, watermark)


        return watermark_out, ber

def log_gradients(model, name="Model"):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm