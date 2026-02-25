import torch
import os
import  tqdm
import hashlib
import numpy as np
import random
from configs.cf_watermark.cf_hufunet import epoch_attack
from networks.cnn import CnnModel
from networks.resnet18_two_linear import ResNet18TwoLinear
import  gc
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint
from util.metric import Metric
from copy import deepcopy
from torch.utils.data import DataLoader
from util.util import Random, TrainModel, Database
from watermark.watermark import Watermark
from networks.hufu_net import HufuNet
from util.hufu_func import Hufu_func
#,   #HufuNetLoss



class HUFUNET(Watermark):
    def __init__(self):
        super().__init__()
        self.hufu=HufuNet()

    @staticmethod
    def get_name():
        return "HUFUNET"
    
    def keygen(self, watermark_size, selected_weights_size):
        pass

    def embed(self, initial_model,  testloader, trainloader,  config) -> object:
        self.hufu=self.hufu.to(config["device"])        
        hufu_path=config["save_path_hufu_original"]
        print("hufu will be loaded from: ", hufu_path)
    
        train_loader_hufu, val_loader_hufu, test_loader_hufu = Hufu_func.create_data_loaders_hufu(
            batch_size=config['batch_size_hufu'],
            validation_split=config['validation_split']
        )


        if os.path.exists(hufu_path) and config["load_model"]:
            checkpoint=torch.load(hufu_path, config["device"],  weights_only=False)
            hufu_orig = checkpoint['supplementary']['model']  # Gets your complete model ✅
            best_loss_hufu = checkpoint['supplementary']['best_loss']  # Gets your best loss ✅
            mse_hufu = checkpoint['supplementary']['MSE']
    
    
        else:
    
            hufu_orig, best_loss_hufu=Hufu_func.train_hufu(self.hufu ,train_loader_hufu, val_loader_hufu, test_loader_hufu, config ,is_fine_tune=False, best_loss=float('inf'))
        mse = Metric.calculate_mse_(hufu_orig, train_loader_hufu, config["device"])
    
        print(f"mse_value_hufu_before_training: {mse}")
        hufu=deepcopy(hufu_orig).to(config["device"])
        model_wat = deepcopy(initial_model).to(config["device"])
        # best_model_wat=deepcopy(initial_model).to(config["device"])
        # best_model_hufu=deepcopy(hufu_orig).to(config["device"])
    
        criterion =config["criterion"]
        # hufunet_loss = HufuNetLoss(gamma1=3.0, gamma2=1.5, gamma3=1.0, lambda_reg=0.1)
        optimizer_cifar = torch.optim.Adam(model_wat.parameters(), lr=config["lr"])
    
        print("Starting training...")
    
    
    
        selected_indexes=Hufu_func.select_indexes(hufu, model_wat)
        print("Selected indexes: ", selected_indexes.size())
        total = sum(p.numel() for p in hufu.encoder.parameters())
        trainable = sum(p.numel() for p in hufu.encoder.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    
        total = sum(p.numel() for p in model_wat.parameters())
        trainable = sum(p.numel() for p in model_wat.parameters() if p.requires_grad)
        print(f"Total parameters model: {total:,}")
        print(f"Trainable parameters: {trainable:,}")


        best_mse=float('inf')
        best_accuracy=0
        early_stop=0
        # constraint_satisfied_epochs = 0  # Track consecutive epochs satisfying constraints



        for epoch in range(config["epochs"]):
            # epoch_metrics = {'beta1': [], 'beta2': [], 'beta3': [], 'base_loss': [], 'total_loss': []}
            loss_avg = 0
            model_wat = Hufu_func.embed_encoder_in_model(hufu, model_wat,selected_indexes)
            # test
            pp=torch.cat([p.data.flatten() for p in model_wat.parameters()])
            pp2=torch.cat([p.data.flatten() for p in hufu.encoder.parameters()])
            my_sum=0
            for i , idx in enumerate (selected_indexes):
                # print(f"Index {idx}: {pp[idx].item()} hufu  {pp2[i]}")
                if pp[idx]==pp2[i]:
                    my_sum+=1
    
            if my_sum==len(selected_indexes):
                print(f"epoch:{epoch+1} embedding hufu in model was ok")
            epoch_loss = 0.0
            correct = 0
            total = 0
    
            loop = tqdm.tqdm(trainloader, leave=True)
    
    
            for batch_idx, (inputs, labels) in enumerate(loop):
                    model_wat.train()
                    inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
    
                    # Zero gradients
                    optimizer_cifar.zero_grad()
    
                    # Forward pass
                    outputs = model_wat(inputs)
                    # print(f"outputs size: {outputs.size()}")
    
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()
                    optimizer_cifar.step()
                    epoch_loss += loss.item()
                    loop.set_description(f"watermark model training Epoch [{epoch + 1}/{config['epochs']}]")
                    loop.set_postfix(
                        loss=loss.item(), avgloss=epoch_loss/(batch_idx+1),
                    )
            loss_avg+=(epoch_loss/len(trainloader))
            print(f"watermark model training Epoch [{epoch + 1}/{config['epochs']}] loss: {loss_avg:.4f}")
            #

            hufu_model=Hufu_func.extract_weight_from_model(selected_indexes,model_wat,hufu )
            pp=torch.cat([p.data.flatten() for p in model_wat.parameters()])
            pp2=torch.cat([p.data.flatten() for p in hufu_model.encoder.parameters()])
            my_sum=0
            for i , idx in enumerate (selected_indexes):
                # print(f"Index {idx}: {pp[idx].item()} hufu  {pp2[i]}")
                if pp[idx]==pp2[i]:
                    my_sum+=1
    
            if my_sum==len(selected_indexes):
                print(f"epoch:{epoch+1} embedding model in hufu was ok")
    
    
            mse_value = Metric.calculate_mse_(hufu_model, train_loader_hufu, config["device"])
            print(f"mse_value_hufu_after_training: {mse_value}")
    
            if mse_value>0.004 and epoch<config["epochs"]-1 :
                hufu_model, best_loss_hufu=Hufu_func.train_hufu(hufu_model ,train_loader_hufu,val_loader_hufu, test_loader_hufu,config, is_fine_tune=True, best_loss=best_loss_hufu )
            else:
                with torch.no_grad():  # No gradients needed
                    model_wat.eval()
                    for images, labels in testloader:
                        images, labels = images.to(config["device"]), labels.to(config["device"])
                        outputs = model_wat(images)
                        _, predicted = torch.max(outputs, 1)  # Get class index with highest probability
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()  # Count correct predictions
                accuracy = 100 * correct / total  # Compute accuracy percentage

                best_model_wat = deepcopy(model_wat)
                best_model_hufu = deepcopy(hufu_model)

                print("\n" + "=" * 80)
                print("FINAL EVALUATION Model")
                print("=" * 80)

                print(
                    f"Watermark_model: Epoch [{epoch + 1}/{config["epochs"]}], Test Accuracy: {accuracy:.2f}%, Classification Loss: {loss_avg:.4f}, "
                    f"hufu Loss: {best_loss_hufu:.4f}, Hufu MSE: {mse_value:.4f}")
                supplementary = {'model': best_model_wat, 'watermark': best_model_hufu.encoder,
                                 'selected_indexes': selected_indexes, 'MSE': mse_value}

                TrainModel.save_model(deepcopy(best_model_wat), accuracy, epoch + 1, config['save_path'], supplementary)
                print("Final watermarked model saved!")
                supplementary_hufu = {'model': best_model_hufu,

                                 'MSE': mse_value
                                 }
                TrainModel.save_model(deepcopy(best_model_hufu), _, epoch,
                                      config['save_path_hufu_finetune'],
                                      supplementary_hufu)
                break
                
                
          
    
    
    
    
            with torch.no_grad():  # No gradients needed
                    model_wat.eval()
                    for images, labels in testloader:
                        images, labels = images.to(config["device"]), labels.to(config["device"])
                        outputs = model_wat(images)
                        _, predicted = torch.max(outputs, 1)  # Get class index with highest probability
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()  # Count correct predictions
            accuracy = 100 * correct / total  # Compute accuracy percentage
            print("\n" + "=" * 80)
            print("FINAL EVALUATION Model")
            print("=" * 80)
    
            print(
                    f"watermark model: Epoch [{epoch + 1}/{config["epochs"]}], Test Accuracy: {accuracy:.2f}%, Classification Loss: {loss_avg:.4f}, "
                    f"hufu Loss: {best_loss_hufu:.4f}, Hufu MSE: {mse_value:.4f}")
            early_stop+=1
    
            if (accuracy>85 or accuracy>best_accuracy) and mse_value<best_mse :
                # early_stop=0
                best_accuracy = accuracy
                best_mse = mse_value
                best_model_wat=deepcopy(model_wat)
                best_model_hufu=deepcopy(hufu_model)

                supplementary = {'model': best_model_wat,  'watermark': best_model_hufu.encoder,  'selected_indexes': selected_indexes      }
    
                TrainModel.save_model(deepcopy(best_model_hufu), accuracy, epoch+1, config['save_path'], supplementary)
                print("watermarked model saved!")
            # elif early_stop>20:
            #     break
    
    
    
    
    
    
    
        mse_before, mse_after, mse_non_wm=self.extract(deepcopy(best_model_wat), deepcopy(best_model_hufu), selected_indexes, train_loader_hufu, config)
        print(f"mse_value before extraction {mse_before:.6f}")
        print(f"mse_value after extraction watermarked model {mse_after:.6f}")
        print(f"mse_value after extraction non watermarked {mse_non_wm:.6f}")
        Hufu_func.test_hufu(hufu_model, hufu_orig)
        return model_wat, selected_indexes
    
    
    
    
    def extract(self,model_watermark, hufu, selected_indexes, train_loader_hufu, config):
        mse_before=Metric.calculate_mse_(hufu, train_loader_hufu, config["device"])

    
        newhufu=Hufu_func.extract_weight_from_model( selected_indexes, model_watermark, hufu,)
        mse_after = Metric.calculate_mse_(newhufu, train_loader_hufu, config["device"])

        mse_non_wm=-1
        # print(f"mse_value after extraction non watermarked {mse_non_wm:.6f}")
        return mse_before, mse_after, mse_non_wm
    
    
    
    
    
    
    




