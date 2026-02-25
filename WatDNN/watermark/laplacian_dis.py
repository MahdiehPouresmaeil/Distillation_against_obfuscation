from watermark.watermark  import Watermark
import rsa
import torch
import random
from torch import optim
import torch.nn.functional as F
from copy import deepcopy
from util.util import  TrainModel, Random
from torch.nn import BCELoss
from tqdm import tqdm
import numpy as np
import  sys
import math
from torch.distributions import Laplace

class Laplacian(Watermark):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name():
        return "LAPLACIAN"

    def keygen(self,watermark_size):
        # Generate the watermark
        watermark = torch.tensor(Random.get_rand_bits(watermark_size, 0., 1.)).cuda()
        wat_signs = torch.tensor([-1 if i == 0 else i for i in watermark], dtype=torch.float)
        return wat_signs, watermark

        

    def embed(self, init_model, test_loader, train_loader, config) -> object:

        watermark_sig, watermark=self.keygen(config["watermark_size"])  #the watermark message in {1,+1} with size 256

        gamma=1* (config["Variance"]/np.sqrt(2))

        pseudo_seq=self.generate_laplacian_pseudo_sequence(s=config['spreading_factor'], l=config["watermark_size"], loc=0.0, scale=gamma, seed=None, device=config["device"])


        weight_seq=self.generate_weight_sequence( watermark_signs=watermark_sig, Pseudo_seq=pseudo_seq, S=config['spreading_factor'])
        print(f"freezed weight sequence: {weight_seq}")


        model = deepcopy(init_model)

        selected_index=self.replace_weights(model=model, layer_name=config["layer_name"], weight_sequence=weight_seq)
        print(f"sorted Selected indices: {sorted(selected_index)}")
        print(f"Selected indices: {selected_index}")




        # Loss and optimizer
        criterion = config["criterion"]
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = TrainModel.get_scheduler(optimizer, config)

        ber_ = l_wat = l_global = 1
        best_acc=0



        for epoch in range(config["epochs"]):
            train_loss = correct = total = 0
            i=1
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (inputs, targets) in enumerate(loop):

                # Move tensors to the configured device
                
                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                optimizer.zero_grad()
                outputs = model(inputs)


                # Compute the loss
                # λ, control the trade of between WM embedding and Training
                l_main_task = criterion(outputs, targets)
                l_global = l_main_task


                train_loss += l_main_task.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()



                l_global.backward()




                optimizer.step()



                self.replace_weights(model=model, layer_name=config["layer_name"],
                                     weight_sequence=weight_seq, selected_indices=selected_index)



                wat_and_weight, ber = self._extract(model,  watermark, selected_index, pseudo_seq,config)


                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"
                                               f"/{total}]",
                                 ber=f"{ber:1.3f}"
                                 )
            scheduler.step()

            if (epoch + 1) % config["epoch_check"] == 0:
                with torch.no_grad():
            
                    wat_and_weight, ber = self._extract(model,  watermark, selected_index, pseudo_seq,config)
                    acc = TrainModel.evaluate(model, test_loader, config)
                    print(
                        f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber:1.3f}---acc: {acc}")
                    print(f"extracted weights: {wat_and_weight[1]} ")
                    print(f"weights: {weight_seq} ")

                if acc > best_acc and epoch+1>=30:
                    print("saving!")
                    supplementary = {'model': model,  'watermark': watermark, 'ber': ber,
                                     "layer_name": config["layer_name"], "selected_index": selected_index, "pseudo_seq": pseudo_seq, "weight_seq": weight_seq, "accuracy": acc}

                    self.save(path=config['save_path'], supplementary=supplementary)
                    print("model saved!")
                    break

        return model, ber

    def extract(self, model_watermarked, supp , config):
        return self._extract(model_watermarked, supp["watermark"],supp["selected_index"], supp["pseudo_seq"],config)

    def _extract(self, model, watermark,selected_indices,pseudo_seq, config):


        weight_extract = self.pick_weights_by_indices(model, config['layer_name'], selected_indices)
        watermark_extract,_ = self.calculate_watermark_bits_detailed(weight_extract, pseudo_seq, config["spreading_factor"])


        ber= self._get_ber(watermark_extract, watermark)

        
        
        

        return  [watermark_extract,weight_extract] , ber

    def generate_laplacian_pseudo_sequence(self, s, l, loc=0.0, scale=1.0, seed=None, device="cpu"):
        """
        Generate a pseudo-random sequence with Laplacian distribution.

        Args:
            s (int): Spreading factor
            l (int): Size of watermark
            loc (float): Location parameter (mean) of the Laplacian distribution. Default: 0.0
            scale (float): Scale parameter (diversity) of the Laplacian distribution. Default: 1.0
            seed (int, optional): Random seed for reproducibility. Default: None

        Returns:
            torch.Tensor: Pseudo sequence of size (s * l,) with Laplacian distribution

        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Total sequence size
        total_size = s * l

        # Generate Laplacian distributed sequence using PyTorch
        # PyTorch doesn't have direct Laplacian, so we use the relationship:
        # If U ~ Uniform(-0.5, 0.5), then X = μ - b * sign(U) * ln(1 - 2|U|) ~ Laplace(μ, b)

        # Method 1: Using torch.distributions (recommended)

        laplace_dist = Laplace(loc=torch.tensor(loc), scale=torch.tensor(scale))
        sequence = laplace_dist.sample((total_size,))

        return sequence.to(device)

    def generate_weight_sequence(self, watermark_signs, Pseudo_seq, S):

        """
        Generate a weight sequence with spreading factor S.
        """

        l = len(watermark_signs)  # Length of user sequence
        n = len(Pseudo_seq)  # Length of spreading sequence

        # Check if spreading sequence is long enough
        if n < l * S:
            raise ValueError(f"Spreading sequence length ({n}) must be >= l*S ({l * S})")

        # Initialize watermark sequence
        weight_seq = torch.zeros(l * S, dtype=torch.float32)

        # Apply spreading formula: wm_j = u_i · s_j
        for i in range(1, l + 1):  # i ∈ [1, l]
            start_idx = (i - 1) * S  # (i-1)S (0-indexed)
            end_idx = i * S  # iS (0-indexed)

            # j ∈ {(i-1)S + 1, ..., iS} in 1-indexed
            # j ∈ {(i-1)S, ..., iS-1} in 0-indexed
            for j in range(start_idx, end_idx):
                weight_seq[j] = watermark_signs[i - 1] * Pseudo_seq[j]  # u_i * s_j (adjusting for 0-indexing)

        return weight_seq

    @staticmethod
    def replace_weights( model, layer_name, weight_sequence, selected_indices=None):
        """
        Replace n random weights with sequence values and freeze them during training
        """
        # Convert weight sequence to tensor

        n = len(weight_sequence)

        # Get the target layer
        target_layer = None
        for name, weights in model.named_parameters():
            if name == layer_name:

                # print(name)

                target_layer = weights
                # print(target_layer.shape)
                break
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        # Get the weight tensor
        weight_tensor = target_layer.data
        original_shape = weight_tensor.shape
        # print(original_shape)

        # Flatten for easier indexing
        flat_weights = weight_tensor.flatten()
        total_weights = flat_weights.numel()

        if n > total_weights:
            raise ValueError(f"Sequence length {n} exceeds total weights {total_weights}")


        if selected_indices is None:
            # Randomly select n indices
            selected_indices = random.sample(range(total_weights), n)

        # Replace selected weights with sequence values
        for i, idx in enumerate(selected_indices):
            flat_weights[idx] = weight_sequence[i]

        # Reshape back to original shape
        target_layer.data = flat_weights.reshape(original_shape)

        # Define the freeze hook function
        # def freeze_selected_weights(grad):
        #     """
        #     Hook function to freeze selected weights during backpropagation
        #     """


        # Return information about the replacement
        return    selected_indices


    # def freeze_weights(self, model, layer_name, selected_indices):
    #     for name, weights in model.named_parameters():
    #         if name == layer_name:
    #
    #             print(f"weights {weights}")
    #             print(f"weights.grad  {weights.grad}")
    #             shape_grad=weights.grad.shape
    #             print(shape_grad)
    #             flat_grad=weights.grad.flatten()
    #             print(flat_grad[:10])
    #             for idx in selected_indices:
    #                 flat_grad[idx] = 0.0
    #             print(flat_grad[:10])
    #             weights.grad = flat_grad.reshape(shape_grad)
    #             print(weights.grad)



    @staticmethod
    def pick_weights_by_indices(model, layer_name, selected_indices):
        """
        Pick weights from a specific layer based on selected indices

        Args:
            model: PyTorch model
            layer_name: name of the target layer
            selected_indices: list of indices to pick weights from

        Returns:
            torch.Tensor: selected weights
        """
        # Get the target layer
        target_layer = None

        for name, weights in model.named_parameters():
            if name == layer_name:

                target_layer = weights
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        # Get the weight tensor and flatten it
        weight_tensor = target_layer.data
        flat_weights = weight_tensor.flatten()

        # Pick weights based on selected indices
        selected_weights = flat_weights[selected_indices]

        return selected_weights

    @staticmethod
    def calculate_watermark_bits_detailed(selected_weights, spreading_sequence, S):
        """
        Calculate watermark bits with detailed intermediate results
        """


        # Calculate watermark length
        watermark_length = len(spreading_sequence) // S

        if len(spreading_sequence) != watermark_length * S:
            raise ValueError(f"Spreading sequence length ({len(spreading_sequence)}) must be divisible by S ({S})")

        if len(selected_weights) < len(spreading_sequence):
            raise ValueError(f"Need at least {len(spreading_sequence)} selected weights, got {len(selected_weights)}")

        # Initialize results
        watermark_bits = torch.zeros(watermark_length, dtype=torch.int32)
        correlation_sums = torch.zeros(watermark_length, dtype=torch.float32)


        # Calculate each watermark bit
        for i in range(1, watermark_length + 1):  # i from 1 to watermark_length
            # Calculate range: j from (i-1)*S+1 to i*S
            start_idx = (i - 1) * S  # Convert to 0-based indexing
            end_idx = i * S  # Exclusive end for Python slicing

            # Get the corresponding spreading sequence and weights
            s_slice = spreading_sequence[start_idx:end_idx]
            w_slice = selected_weights[start_idx:end_idx]

            # Calculate sum of s_j * w_j
            correlation_sum = torch.sum(s_slice * w_slice)
            correlation_sums[i - 1] = correlation_sum

            # Set watermark bit: 1 if sum >= 0, otherwise 0
            bit_value = 1 if correlation_sum >= 0 else 0
            watermark_bits[i - 1] = bit_value



        return  watermark_bits, correlation_sums
