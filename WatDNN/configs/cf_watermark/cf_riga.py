from copy import deepcopy

from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict

#********************************************************************
#************** watermarking MLP architecture ***********************
#********************************************************************
method_wat = "riga"
watermark_size = 256
epochs_embed = 1000
layer_name = "fc1.weight"
epoch_check = 10
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) +\
                  "_epc" + str(epoch_check)
cf_mlp_embed = {"method_wat": method_wat,
                "configuration": cf_mlp_dict,
                "database": cf_mlp_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                "batch_size": 128,
                "lambda_1": 0.01,
                "lambda_2": 0.1,

                "layer_name": layer_name,
                "lr": 1e-4,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_mlp_dict["architecture"],

                "path_model": cf_mlp_dict["save_path"],
                "save_path": "results/watermarked_models/uchida/" + cf_mlp_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",

                "momentum": 0,
                "milestones": [100, 150],
                "gamma": 0,
                "criterion": cf_mlp_dict["criterion"],
                "device": cf_mlp_dict["device"],
                }


#### *********************************Fine tuning attack************************************
epoch_attack = 2
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + ".pth"
cf_mlp_attack_ft = {}

### *********************************Pruning attack****************************************
cf_mlp_attack_pr = {}

### *********************************Overwriting attack************************************
epoch_attack = 1000
watermark_size = 256
epoch_check = 2
layer_name = ["fc1"]
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)
cf_mlp_attack_ow = {}

### *********************************PIA attack**********************************************
cf_non_watermarked = deepcopy(cf_mlp_dict)
cf_non_watermarked["epochs"] = 2
cf_non_watermarked["show_acc_epoch"] = False
cf_watermarked = deepcopy(cf_mlp_embed)
cf_watermarked["epochs"] = 10
cf_watermarked["show_acc_epoch"] = False
nb_examples = 800
save_path_attack = "_l" + layer_name[0] + "_wat" + str(
    cf_watermarked["watermark_size"]) + "_ep" + str(cf_non_watermarked["epochs"]) + "_nb_examples" + str(nb_examples)
cf_mlp_attack_pia = {}

### ****************************Dummy neurons attack***************************************
layer_name="fc2"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_mlp_attack_dummy_neurons = {
    "configuration": cf_mlp_dict,
    "database": cf_mlp_dict["database"],
    "path_model": cf_mlp_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_mlp_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### **********************Distillation attack************************
layer_name="fc2"
attack_type="logits"
epoch_attack = 20
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_mlp_attack_distillation = {
    "configuration": cf_mlp_dict,
    "database": cf_mlp_dict["database"],
    "path_model": cf_mlp_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_mlp_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}

#********************************************************************
#************** watermarking CNN architecture ***********************
#********************************************************************
watermark_size = 256
epochs_embed = 1000
layer_name = "conv4.weight"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_cnn_embed = {"method_wat": method_wat,
                "configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                # Latent space parameters
                "batch_size": cf_cnn_dict["batch_size"] // 2,
                "mean": 0,
                "std": 1,
                "n_features": 512 // 1,
                "n_features_layer": 512,
                "lambda": 1e-0,
                "layer_name": layer_name,

                "lr": 1e-3,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict['architecture'],
                "momentum": 0,
                "milestones": [20, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict["criterion"],
                "device": cf_cnn_dict["device"],

                "path_model": cf_cnn_dict["save_path"],
                "save_path": "results/watermarked_models/riga/" + cf_cnn_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                "lambda_1": 0.01,
                "lambda_2": 0.1,
                }


### **************** Fine tuning attack *******************************
epoch_attack = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_cnn_attack_ft = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed['save_path'],
                    "epochs": epoch_attack,
                    "watermark_size": watermark_size,
                    "layer_name": layer_name,

                    "lr": 1e-4,
                    "wd": 1e-5,

                    "scheduler": "MultiStepLR",
                    "opt": "Adam",
                    "architecture": cf_cnn_dict["architecture"],
                    "save_path": "results/attacks/finetuning/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0.9,
                    "milestones": [150, 200],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],
                    "save_fig_path": "results/attacks/finetuning/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_Dither_CNN",
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

### **************** Pruning attack *******************************
cf_cnn_attack_pr = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,
                    "architecture": cf_cnn_dict['architecture'],
                    "save_path": "results/attacks/pruning/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    }

### *************** Overwriting attack ****************************
layer_name = "conv4.weight"
epoch_attack = 20
watermark_size = 256
epoch_check = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)
cf_cnn_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack ,
                    "epoch_check": epoch_check,

                    "lambda_1": 1,

                    "layer_name": layer_name,
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_cnn_dict["architecture"],

                    "path_model": cf_cnn_embed["save_path"],
                    "save_path": "results/attacks/overwriting/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_Dither_cnn" + cf_cnn_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False,
                    "alpha":1,
                    "beta":1,

                    "lambda_1": 0.01,
                    "lambda_2": 0.1,
                    }

cf_cnn_attack_pia = {}

### ****************************Dummy neurons attack***************************************
layer_name="fc1"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_cnn_attack_dummy_neurons = {
    "configuration": cf_cnn_dict,
    "database": cf_cnn_dict["database"],
    "path_model": cf_cnn_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_cnn_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_cnn_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### **********************Distillation attack************************
layer_name="conv4"
attack_type="logits"
epoch_attack = 150
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_cnn_attack_distillation = {

"method_wat": method_wat,
    "configuration": cf_cnn_dict,
    "database": cf_cnn_dict["database"],
    "path_model": cf_cnn_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_cnn_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_cnn_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}

cf_cnn_attack_ambiguity= {}

#********************************************************************
#************** watermarking Resnet18 architecture ******************
#********************************************************************
watermark_size = 256
epochs_embed = 1000
# layer_name = "base.fc.0.weight"
# layer_name="base.layer3.0.conv2.weight"
layer_name="base.layer4.1.conv2.weight"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed)
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)

cf_resnet18_embed = {"method_wat": method_wat,
                    "configuration": cf_resnet18_dict,
                     "database": cf_resnet18_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,



                      "lambda_1": 0.01,
                    "lambda_2": 0.1,

                     "layer_name": layer_name,
                     "lr": 1e-4,
                     "wd": 5e-5,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_resnet18_dict['architecture'],

                     "save_path": "results/watermarked_models/" + method_wat + "/" + cf_resnet18_dict[
                         'architecture'].lower() + "/" + save_path_embed + ".pth",

                     # relu but change in mlp cd
                     "path_model": cf_resnet18_dict["save_path"],
                     "momentum": 0,
                     "milestones": [100, 150],
                     "gamma": 0,
                     "criterion": cf_resnet18_dict["criterion"],
                     "device": cf_resnet18_dict["device"],
                     }

### ****************************Fine tuning attack***************************************
epoch_attack = 10
save_path_attack = layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_resnet18_attack_ft = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed['save_path'],
                         "epochs": epoch_attack,
                         "watermark_size": watermark_size,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 1e-5,
                         "scheduler": "MultiStepLR",
                         "opt": "Adam",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/finetuning/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [150, 200],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_res_encrypt_resnet18",
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

### ****************************Pruning attack***************************************
cf_resnet18_attack_pr = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"],
                         "watermark_size": watermark_size,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/pruning/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         }

### ****************************Overwriting attack***************************************
watermark_size = 256
# layer_name = "base.fc.0.weight"
# layer_name="base.layer3.0.conv2.weight"
layer_name="base.layer2.0.conv2.weight"
epoch_attack = 100
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)

cf_resnet18_attack_ow = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         "lambda_1": 1,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 1e-5,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/overwriting/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "lambda_2": 0.1,

                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_res_encrypt" + cf_resnet18_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

### ****************************PIA attack***************************************
cf_resnet18_attack_pia = {}

### ****************************Dummy neurons attack***************************************
layer_name="base.fc.0"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_resnet18_attack_dummy_neurons = {
    "configuration": cf_resnet18_dict,
    "database": cf_resnet18_dict["database"],
    "path_model": cf_resnet18_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_resnet18_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_resnet18_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### **********************Distillation attack************************
layer_name="linear"
attack_type="logits"
epoch_attack =150
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_resnet18_attack_distillation = {
    "configuration": cf_resnet18_dict,
    "database": cf_resnet18_dict["database"],
    "path_model": cf_resnet18_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_resnet18_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_resnet18_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack,
"method_wat": method_wat,

}
