import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader
from networks.hufu_net import HufuNet
import torch.nn as nn
import  tqdm
from util.metric import Metric
from util.util import TrainModel
from copy import deepcopy
import numpy as np
import hashlib
import random


class Hufu_func:
    @staticmethod
    def create_data_loaders_hufu(batch_size=128, validation_split=0.1):
        """
        Create optimized data loaders for MNIST
        """
        # Optimal transforms for MNIST autoencoders
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Note: No normalization for autoencoders - keep in [0,1] range
        ])

        # Load full training dataset
        full_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        # Split into train/validation
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Test dataset
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader
    
    @staticmethod
    def train_hufu(model, train_loader_hufu, val_loader_hufu, test_loader_hufu, config, is_fine_tune,
                   best_loss=float('inf')) -> object:
        print("Training HufuNet")
        # metrics_calculator = AutoencoderMetrics(config["device"])
        best_model = HufuNet().to(config["device"])
        if is_fine_tune:
            for param in model.decoder.parameters():
                param.requires_grad = False
            epochs = config["epoch_hufu_finetune"]
            print("decoder freezed")
        else:
            epochs = config['epoch_hufu']
            print(f"epochs {epochs}")

        model.to(config["device"])
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_hufu"])
        best_val_loss = float('inf')
        patience_counter = 0

        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            loop = tqdm.tqdm(train_loader_hufu, leave=True)

            for batch_idx, (images, _) in enumerate(loop):
                images = images.to(config["device"])
                optimizer.zero_grad()
                _, output = model(images)
                loss = criterion(output, images)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(train_loader_hufu)
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data, _ in val_loader_hufu:
                    data = data.to(config["device"])
                    _, reconstructed = model(data)
                    loss = criterion(reconstructed, data)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader_hufu)
            val_losses.append(avg_val_loss)
            if epoch % 10 == 0 or epoch == config['epochs'] - 1:
                train_metrics = Metric.evaluate_model(model, train_loader_hufu, config["device"])
                val_metrics = Metric.evaluate_model(model, val_loader_hufu, config["device"])
                train_metrics_history.append(train_metrics)
                val_metrics_history.append(val_metrics)
                print(f'Epoch [{epoch + 1}/{epochs}]')

                print(f'  Train - Loss: {avg_train_loss:.6f},'
                      f' MSE: {train_metrics["mse"]:.6f}, '
                      f'PSNR: {train_metrics["psnr"]:.2f}'
                      )

                print(f'  Val   - Loss: {avg_val_loss:.6f}, MSE: {val_metrics["mse"]:.6f}, '
                      f'PSNR: {val_metrics["psnr"]:.2f},'
                      )
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
            else:
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.6f}, '
                      f'Val Loss: {avg_val_loss:.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                supplementary = {'model': model,
                                 'best_loss': best_loss,
                                 'MSE': train_metrics["mse"]
                                 }
                # Save best model
                if is_fine_tune:
                    TrainModel.save_model(deepcopy(model), _, epoch,
                                          config['save_path_hufu_finetune'],
                                          supplementary)
                    best_model = deepcopy(model)
                    print(f"HufuNet model finetune saved at epoch {epoch}!")
                else:
                    TrainModel.save_model(deepcopy(model),_, epoch,
                                          config['save_path_hufu_original'],
                                          supplementary)
                    best_model = deepcopy(model)
                    print(f"HufuNet model original saved at epoch {epoch}!")

            else:
                patience_counter += 1

            if patience_counter >= config['early_stopping_patience']:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

                # Load best model

        print("\n" + "=" * 80)
        print("FINAL EVALUATION HUFU")
        print("=" * 80)
        final_train_metrics = Metric.evaluate_model(best_model, train_loader_hufu, config["device"])
        final_val_metrics = Metric.evaluate_model(best_model, val_loader_hufu, config["device"])
        final_test_metrics = Metric.evaluate_model(best_model, test_loader_hufu, config["device"])
        print("Final Metrics:")
        print(f"Train - MSE: {final_train_metrics['mse']:.6f}, MAE: {final_train_metrics['mae']:.6f}, "
              f"PSNR: {final_train_metrics['psnr']:.2f}dB,")

        print(f"Val   - MSE: {final_val_metrics['mse']:.6f}, MAE: {final_val_metrics['mae']:.6f}, "
              f"PSNR: {final_val_metrics['psnr']:.2f}dB,")

        print(f"Test  - MSE: {final_test_metrics['mse']:.6f}, MAE: {final_test_metrics['mae']:.6f}, "
              f"PSNR: {final_test_metrics['psnr']:.2f}dB,")

        return best_model, best_val_loss  # , val_losses, train_metrics_history, val_metrics_history

    @staticmethod
    def get_decoder_seed(model):
        # Extract all parameters of the decoder as a single flat tensor
        params = []
        for p in model.decoder.parameters():
            params.append(p.detach().cpu().numpy().flatten())
        flat_params = np.concatenate(params)
        # print(flat_params.size)

        # Convert parameters to a bytestring (e.g., float32 to bytes)
        byte_rep = flat_params.tobytes()
        # print(byte_rep)

        # Hash the byte string (using SHA256 for a stable, unique fingerprint)
        hash_digest = hashlib.sha256(byte_rep).hexdigest()
        # print(hash_digest)
        # Convert hash digest to an integer
        seed = int(hash_digest, 16) % (2 ** 32)  # Use 32-bit seed
        return seed

    @staticmethod
    def select_indexes(hufu, model_wat):
        seed = Hufu_func.get_decoder_seed(hufu)
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        all_params_hufu = torch.cat([p.data.flatten() for p in hufu.encoder.parameters()])
        # print(all_params_hufu.shape)

        all_params_model = torch.cat([p.data.flatten() for p in model_wat.parameters()])
        # print(all_params_model.shape)

        y = all_params_hufu.size(0)

        # Select random indices
        selected_indices = torch.randperm((all_params_model.size(0)))[:y]
        # print(y)
        # print(selected_indices.shape)
        return selected_indices

    @staticmethod
    def embed_encoder_in_model(hufu, model_wat, selected_indices):
        encoder_params_hufu = torch.cat([p.data.flatten() for p in hufu.encoder.parameters()])

        selected_indices = selected_indices.tolist()
        assert len(selected_indices) == len(encoder_params_hufu)

        current_update = 0
        count = 0

        for p in model_wat.parameters():
            numel = p.numel()
            p_flat = p.data.view(-1)

            for i in range(len(selected_indices)):
                flat_index = selected_indices[i]
                if count <= flat_index < count + numel:
                    local_index = flat_index - count
                    with torch.no_grad():
                        p_flat[local_index] = encoder_params_hufu[i]
            count += numel

        return model_wat
    
    @staticmethod
    def extract_weight_from_model(selected_indexes, model_wm, model_hufu):
        all_params_model = torch.cat([p.data.flatten() for p in model_wm.parameters()])

        point = 0

        for p in model_hufu.encoder.parameters():
            numel = p.data.numel()
            # p_flat=p.data.flatten()
            for i in range(numel):
                j = point + i
                p.data.flatten()[i] = all_params_model[selected_indexes[j]]
            point += numel

        return model_hufu

    @staticmethod
    def test_hufu( hufu, hufu_orig):
        decoder1_params = torch.nn.utils.parameters_to_vector(hufu.decoder.parameters())
        decoder2_params = torch.nn.utils.parameters_to_vector(hufu_orig.decoder.parameters())

        difference = torch.norm(decoder1_params - decoder2_params).item()
        print(f"L2 norm of difference between decoders: {difference}")

        encoder1_params = torch.nn.utils.parameters_to_vector(hufu.encoder.parameters())
        encoder2_params = torch.nn.utils.parameters_to_vector(hufu_orig.encoder.parameters())

        difference1 = torch.norm(encoder1_params - encoder2_params).item()
        print(f"L2 norm of difference between encoders: {difference1}")


