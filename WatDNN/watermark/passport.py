import torch.nn.functional as F
from torch import optim
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from copy import deepcopy
import torch
from tqdm import tqdm
from util.util import TrainModel, Random
from watermark.watermark import Watermark

# Global flag to reduce logging spam
_PASSPORT_LOG_FLAGS = {
    'channel_mismatch': False,
    'truncate': False,
    'pad': False,
    'conv_error': False
}
class Passport(Watermark):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name():
        return "Passport"

    def keygen(self, watermark_size):
        """
        Generate a secret key and message.
        """
        watermark = Random.get_rand_bits(watermark_size, 0., 1.)
        watermark = torch.tensor(watermark).reshape(1, watermark_size)
        return watermark

    def embed(self, init_model, test_loader, train_loader, config) -> object:
        device = config["device"]
        architecture = config["architecture"]

        print(f"Loading {architecture} model...")
        original_model = init_model.to(device)

        # Create trigger image
        trigger_image, _ = next(iter(test_loader))
        trigger_image = trigger_image[0].unsqueeze(0).to(device)
        print(f"Trigger image size from the testing set: {trigger_image.shape}")

        # Get target layer info
        target_layer_name = config["layer_name"][0]
        print("Available layers in the model:")
        available_layers = get_graph_node_names(original_model)
        # print(available_layers)
        print(f"Target layer: {target_layer_name}")

        # >>>>>>> UNIFIED PASSPORT CREATION <<<<<<<
        # Create passport activations based on architecture
        passport_activations = self._create_universal_passport(
            original_model, target_layer_name, trigger_image, architecture
        )

        print(f"Applying passport to layer: {target_layer_name}")
        print(f"Passport activation shape: {passport_activations[target_layer_name].shape}")

        # Create passport model with hooks
        passport_model, hooks = self._create_passport_model(
            original_model, passport_activations, [target_layer_name]
        )
        passport_model = passport_model.to(device)

        # >>>>>>> UNIFIED WATERMARK SIZE CALCULATION <<<<<<<
        watermark_size = self._get_universal_watermark_size(
            original_model, target_layer_name, architecture
        )
        watermark = torch.randint(0, 2, (watermark_size,), dtype=torch.float32) * 2 - 1
        watermark = watermark.to(device)

        print(f"Watermark size: {len(watermark)}, sample: {watermark[:min(10, len(watermark))]}...")

        # Training setup
        optimizer = optim.Adam(passport_model.parameters(), lr=config.get("lr", 0.001))
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get("milestones", [20, 60]),
            gamma=config.get("gamma", 0.1)
        )
        criterion = config["criterion"]

        # best_ber = 1.0
        best_acc = 0.0
        best_model = None

        # Training loop
        for epoch in range(config["epochs"]):
            passport_model.train()
            epoch_correct = 0
            epoch_total = 0

            loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1}/{config['epochs']}")

            for batch_idx, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = passport_model(inputs)
                classification_loss = criterion(outputs, labels)

                # Get passport parameters from hook
                hook = hooks[target_layer_name]
                try:
                    # >>>>>>> UNIFIED ALPHA COMPUTATION <<<<<<<
                    alpha, _ = self._get_universal_alpha(hook, inputs.size(0), architecture)

                    # Sign loss to embed watermark in scale factors
                    sign_loss = self._sign_loss(watermark, alpha, gamma=0.1)

                    # Combined loss
                    total_loss = classification_loss + sign_loss

                    # Backward pass
                    total_loss.backward()
                    optimizer.step()

                    # Statistics
                    _, predicted = outputs.max(1)
                    epoch_total += labels.size(0)
                    epoch_correct += predicted.eq(labels).sum().item()
                    ber = self._evaluate_watermark_extraction(
                        hooks[target_layer_name], watermark, architecture
                    )

                    # Update progress bar every 100 batches to reduce spam
                    if batch_idx % 100 == 0:
                        accuracy = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0
                        loop.set_postfix({
                            'l_main_task=': f"{(classification_loss/(batch_idx+1)):4f}",
                            'Loss': f'{total_loss.item():.4f}',
                            'ber ': f"{ber:1.3f}",
                            'Acc': f'{accuracy:.1f}%',
                            'correct_total' : f"[{epoch_correct}/{epoch_total}]",
                            'wat_loss': f'{sign_loss.item():.6f}'
                        })

                except Exception as e:
                    if batch_idx == 0:  # Only print on first batch to avoid spam
                        print(f"Warning in training step: {e}")
                    # Continue with just classification loss
                    classification_loss.backward()
                    optimizer.step()

            scheduler.step()

            # Evaluate watermark extraction every epoch_check epochs
            if (epoch + 1) % config.get("epoch_check", 5) == 0:
                ber = self._evaluate_watermark_extraction(
                    hooks[target_layer_name], watermark, architecture
                )
                train_accuracy = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0

                # Evaluate on test set
                test_accuracy = TrainModel.evaluate(passport_model, test_loader, config)

                print(f"\nEpoch {epoch + 1}: Train Acc = {train_accuracy:.2f}%, "
                      f"Test Acc = {test_accuracy:.2f}%, BER = {ber:.4f}")

                if ber==0.0:
                    # Final evaluation
                    final_test_accuracy = TrainModel.evaluate(passport_model, test_loader, config)
                    final_ber = self._evaluate_watermark_extraction(
                        hooks[target_layer_name], watermark, architecture
                    )
                    print(f"\nFinal Results:")
                    print(f"Test Accuracy: {final_test_accuracy:.2f}%")
                    print(f"Final BER: {final_ber:.4f}")

                    # Save watermarked model
                    supplementary = {
                        "model": passport_model,
                        "hooks": hooks,
                        "layer_name": target_layer_name,
                        'watermark': watermark,
                        'passport_activations': passport_activations,
                        "target_layers": target_layer_name,
                        'test_accuracy': final_test_accuracy,
                        'ber': final_ber,
                        'architecture': architecture  # >>>>>>> NEW: Save architecture info <<<<<<<
                    }
                    self.save(config["save_path"], supplementary)
                    print(f"\nTraining completed! Test Accuracy: {final_test_accuracy:.2f}%, Best BER: {final_ber:.4f}")
                    break





        return passport_model, final_ber

    def _create_universal_passport(self, model, target_layer_name, trigger_image, architecture):
        """>>>>>>> NEW: Create passport activations for any architecture <<<<<<<"""

        print(architecture)

        if architecture in {"ResNet18"}:
            # For ResNet, extract activations from previous layers
            target_layer = self._get_layer_by_name(model, target_layer_name)

            # Check if it's a Linear (FC) layer
            if hasattr(target_layer, 'in_features'):
                # For FC layers, flatten the trigger image
                passport = trigger_image.flatten(1)  # [1, 3*H*W]
                return {target_layer_name: passport}
            else:
                # For Conv layers, extract activations from previous layers
                passport_layers = [target_layer_name]
                model.eval()
                feature_extractor = create_feature_extractor(model, return_nodes=passport_layers)
                with torch.no_grad():
                    passport_activations = feature_extractor(trigger_image)
                return passport_activations

        elif architecture in {"CNN"}:
            # For CNN, create compatible passport based on layer requirements
            target_layer = self._get_layer_by_name(model, target_layer_name)

            if hasattr(target_layer, 'in_channels'):
                required_channels = target_layer.in_channels

                if required_channels == 3:
                    passport = trigger_image
                else:
                    # Create synthetic passport with correct dimensions
                    h, w = trigger_image.shape[2], trigger_image.shape[3]
                    if required_channels <= 64:
                        passport = torch.randn(1, required_channels, h, w, device=trigger_image.device)
                        # Add structure based on original image
                        for c in range(min(3, required_channels)):
                            passport[0, c] = trigger_image[0, c % 3]
                    else:
                        passport = torch.randn(1, required_channels, h, w, device=trigger_image.device)
            else:
                # For fully connected layers
                passport = trigger_image.flatten(1)

            return {target_layer_name: passport}

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def _get_universal_watermark_size(self, model, target_layer_name, architecture):
        """>>>>>>> NEW: Get watermark size for any architecture <<<<<<<"""

        if architecture in {"ResNet18"}:
            layer = self._get_layer_by_name(model, target_layer_name)

            # Check if it's a Linear layer
            if hasattr(layer, 'out_features'):
                return layer.out_features  # For FC layers
            else:
                # For Conv layers, get from weight tensor input channels
                weights = [param for name, param in model.named_parameters()
                           if name == target_layer_name + ".weight"][0]
                return weights.shape[1]  # Input channels

        elif architecture in {"CNN"}:
            # For CNN, get from layer output channels
            layer = self._get_layer_by_name(model, target_layer_name)
            if hasattr(layer, 'out_channels'):
                return layer.out_channels
            elif hasattr(layer, 'out_features'):
                return layer.out_features
            else:
                raise ValueError(f"Cannot determine output size for layer {target_layer_name}")
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def _get_universal_alpha(self, hook, batch_size, architecture):
        """>>>>>>> NEW: Get alpha/beta for any architecture <<<<<<<"""

        if architecture in {"ResNet18"}:
            # ResNet uses specific shape computation
            if hasattr(hook.conv_layer, 'out_features'):
                # For FC layers, just pass batch_size
                alpha, beta = hook.compute_alpha_beta(batch_size)
            else:
                # For Conv layers, use standard ResNet shape
                alpha, beta = hook.compute_alpha_beta((batch_size, 64, 1, 1))
            # alpha, beta = hook.compute_alpha_beta((batch_size, 64, 1, 1))  # Standard ResNet shape
        elif architecture in {"CNN"}:
            # CNN uses batch size computation
            alpha, beta = hook.compute_alpha_beta(batch_size)
        else:
            # Fallback
            alpha, beta = hook.compute_alpha_beta(batch_size)

        return alpha, beta

    def _evaluate_watermark_extraction(self, hook, watermark, architecture):
        """>>>>>>> MODIFIED: Architecture-aware BER evaluation <<<<<<<"""
        try:
            with torch.no_grad():
                alpha, _ = self._get_universal_alpha(hook, 1, architecture)

                _, alpha_signs = self._check_sign_match(alpha, watermark)
                # print(f"DEBUG: alpha_signs sample = {alpha_signs[:10]}")
                print(watermark)

                ber = (1. * (~(alpha_signs == watermark))).mean().item()
                # print(f"DEBUG: BER = {ber}")
                return ber
        except Exception as e:
            print(f"DEBUG: Exception in BER evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1.0  # Worst case BER  # Worst case BER

    def extract(self, classifier, supp):
        """Extract watermark from model"""
        architecture = supp.get('architecture', 'CNN')  # >>>>>>> NEW: Use saved architecture <<<<<<<
        return self._extract(
            classifier,
            supp['watermark'],
            supp['layer_name'],
            supp['passport_activations'],
            supp['target_layers'],
            architecture
        )

    def _extract(self, passport_model, watermark, layer_name, passport_activations, target_layers, architecture):
        """>>>>>>> MODIFIED: Architecture-aware extraction <<<<<<<"""
        passport_model, hooks = self._create_passport_model(
            passport_model, passport_activations, [target_layers]
        )

        passport_model.eval()
        try:
            with torch.no_grad():
                hook = hooks[layer_name]
                alpha, _ = self._get_universal_alpha(hook, 1, architecture)
                sign_accuracy, alpha_signs = self._check_sign_match(alpha, watermark)
                ber = (1. * (~(alpha_signs == watermark))).mean().item()
        except:
            ber = 1.0
            alpha_signs = torch.zeros_like(watermark)

        # Clean up hooks
        for hook in hooks.values():
            hook.remove()

        return alpha_signs.cpu().numpy(), ber

    # >>>>>>> UNCHANGED HELPER METHODS <<<<<<<
    def _get_layer_by_name(self, model, layer_name):
        """Get a layer by its name"""
        layer = model
        for part in layer_name.split('.'):
            layer = getattr(layer, part)
        return layer

    def _create_passport_model(self, base_model, passport_data, target_layers):
        """Create passport model with hooks"""
        passport_config = {}
        for layer_name in target_layers:
            if layer_name in passport_data:
                passport_config[layer_name] = passport_data[layer_name]
            else:
                print(f"Warning: No passport data found for layer {layer_name}")
        return self._apply_passport_to_model(base_model, passport_config)

    @staticmethod
    def _apply_passport_to_model(model, passport_config):
        """Apply passport transformations using hooks"""
        hooks = {}
        model = deepcopy(model)

        for layer_path, passport_activation in passport_config.items():
            layer = model
            for part in layer_path.split('.'):
                layer = getattr(layer, part)
            passport_hook = PassportHook(layer, passport_activation)
            passport_hook.register(layer)
            hooks[layer_path] = passport_hook
        return model, hooks

    @staticmethod
    def _sign_loss(binary_watermark, alpha, gamma=0.1):
        """Compute sign loss to embed binary watermark in scale factor signs"""
        if alpha.dim() > 1:
            alpha = alpha.view(-1)
        binary_watermark = binary_watermark.view(-1)

        min_len = min(len(alpha), len(binary_watermark))
        alpha = alpha[:min_len]
        binary_watermark = binary_watermark[:min_len]

        loss_values = torch.clamp(gamma - alpha * binary_watermark, min=0)
        return torch.sum(loss_values)

    @staticmethod
    def _check_sign_match(alpha, binary_watermark):
        """Check how many signs match between alpha and binary watermark"""
        if alpha.dim() > 1:
            alpha = alpha.view(-1)
        binary_watermark = binary_watermark.view(-1)

        alpha_signs = torch.sign(alpha)
        min_len = min(len(alpha_signs), len(binary_watermark))
        alpha_signs = alpha_signs[:min_len]
        binary_watermark = binary_watermark[:min_len]

        matches = (alpha_signs == binary_watermark).float()
        accuracy = torch.mean(matches).item()
        return accuracy, alpha_signs


class PassportHook:
    """>>>>>>> UNIFIED: Hook that works for both CNN and ResNet18 <<<<<<<"""

    def __init__(self, conv_layer, passport_activation):
        self.conv_layer = conv_layer
        self.passport_activation = passport_activation.clone()
        self.hook = None

        if len(self.passport_activation.shape) == 3:
            self.passport_activation = self.passport_activation.unsqueeze(0)

    def compute_alpha_beta(self, output_shape):
        """>>>>>>> UNIFIED: Compute α and β for any layer type <<<<<<<"""
        global _PASSPORT_LOG_FLAGS

        if hasattr(self.conv_layer, 'weight'):
            conv_weights = self.conv_layer.weight

            # Check if this is a Linear (FC) layer
            if conv_weights.dim() == 2:  # Linear layer: [out_features, in_features]
                out_channels = conv_weights.shape[0]
                expected_in_channels = conv_weights.shape[1]

                # Ensure passport is 2D for Linear layers
                if self.passport_activation.dim() == 4:
                    passport_flat = self.passport_activation.flatten(1)  # [1, C*H*W]
                else:
                    passport_flat = self.passport_activation  # Already [1, features]

                passport_in_channels = passport_flat.shape[1]

                # Log mismatch only once
                if passport_in_channels != expected_in_channels and not _PASSPORT_LOG_FLAGS['channel_mismatch']:
                    print(f"Passport feature adaptation: {passport_in_channels} → {expected_in_channels}")
                    _PASSPORT_LOG_FLAGS['channel_mismatch'] = True

                # Handle feature mismatch
                if passport_in_channels != expected_in_channels:
                    if passport_in_channels > expected_in_channels:
                        passport_to_use = passport_flat[:, :expected_in_channels]
                    else:
                        repeat_factor = (expected_in_channels + passport_in_channels - 1) // passport_in_channels
                        passport_expanded = passport_flat.repeat(1, repeat_factor)
                        passport_to_use = passport_expanded[:, :expected_in_channels]
                else:
                    passport_to_use = passport_flat

                # Apply linear transformation: passport @ W^T
                conv_result = F.linear(passport_to_use, conv_weights)  # [1, out_channels]

                alpha = conv_result.mean(dim=0, keepdim=True)  # [1, out_channels]
                beta = conv_result.mean(dim=0, keepdim=True) * 0.1  # [1, out_channels]

                return alpha, beta

            # Otherwise it's a Conv layer (weights.dim() == 4)
            else:
                out_channels = conv_weights.shape[0]
                expected_in_channels = conv_weights.shape[1]
                passport_in_channels = self.passport_activation.shape[1]

                # Log mismatch only once globally
                if passport_in_channels != expected_in_channels and not _PASSPORT_LOG_FLAGS['channel_mismatch']:
                    print(f"Passport channel adaptation: {passport_in_channels} → {expected_in_channels}")
                    _PASSPORT_LOG_FLAGS['channel_mismatch'] = True

                # Handle channel mismatch
                if passport_in_channels != expected_in_channels:
                    if passport_in_channels > expected_in_channels:
                        passport_to_use = self.passport_activation[:, :expected_in_channels, :, :]
                    else:
                        repeat_factor = (expected_in_channels + passport_in_channels - 1) // passport_in_channels
                        passport_expanded = self.passport_activation.repeat(1, repeat_factor, 1, 1)
                        passport_to_use = passport_expanded[:, :expected_in_channels, :, :]
                else:
                    passport_to_use = self.passport_activation

                try:
                    conv_result = F.conv2d(
                        passport_to_use,
                        conv_weights,
                        bias=None,
                        stride=getattr(self.conv_layer, 'stride', 1),
                        padding=getattr(self.conv_layer, 'padding', 0),
                        dilation=getattr(self.conv_layer, 'dilation', 1)
                    )

                    alpha = torch.mean(conv_result, dim=(2, 3), keepdim=True)
                    beta = torch.mean(conv_result, dim=(2, 3), keepdim=True) * 0.1
                    return alpha, beta

                except RuntimeError as e:
                    if not _PASSPORT_LOG_FLAGS['conv_error']:
                        print(f"Convolution error, using fallback: {e}")
                        _PASSPORT_LOG_FLAGS['conv_error'] = True

                    alpha = torch.mean(passport_to_use) * torch.ones(1, out_channels, 1, 1,
                                                                     device=self.passport_activation.device)
                    beta = torch.zeros(1, out_channels, 1, 1, device=self.passport_activation.device)
                    return alpha, beta
        else:
            # Fallback for layers without weights
            out_features = getattr(self.conv_layer, 'out_features', 128)
            alpha = torch.randn(1, out_features, device=self.passport_activation.device)
            beta = torch.zeros(1, out_features, device=self.passport_activation.device)
            return alpha, beta

    def hook_fn(self, module, input, output):
        """>>>>>>> UNIFIED: Apply passport transformation for any layer <<<<<<<"""
        try:
            alpha, beta = self.compute_alpha_beta(output.shape)
            alpha = alpha.to(output.device)
            beta = beta.to(output.device)

            # Shape compatibility
            if output.dim() == 4 and alpha.dim() == 4:
                pass  # Already compatible
            elif output.dim() == 4 and alpha.dim() == 2:
                alpha = alpha.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
            elif output.dim() == 2 and alpha.dim() == 4:
                alpha = alpha.squeeze(-1).squeeze(-1)
                beta = beta.squeeze(-1).squeeze(-1)

            transformed_output = alpha * output + beta
            return transformed_output

        except Exception:
            return output  # Silently return original on error

    def register(self, module):
        """Register the hook"""
        self.hook = module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove the hook"""
        if self.hook:
            self.hook.remove()




