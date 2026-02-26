# WatDNN: Turning Distillation against Obfuscation: A Recovery Framework for DNN White-Box Watermarks

## Project Information

| Item | Details |
|------|---------|
| **Project Name** | WatDNN |
| **Full Title** | Turning Distillation against Obfuscation: A Recovery Framework for DNN White-Box Watermarks |
| **Version** | 1.0.0 |
| **Release Date** | February 2026 |
| **License** | MIT |
| **Python Version** | >= 3.10 |
| **PyTorch Version** | >= 2.2.2 |
| **Repository Type** | Research Framework |
| **Primary Focus** | Distillation-based Defense Against Obfuscation Attacks |
| **Watermarking Schemes** | 10 schemes implemented |
| **Obfuscation Attacks** | 5 attack methods |
| **Supported Datasets** | 5 datasets (CIFAR-10/100, MNIST, CelebA, SST-2) |
| **Model Architectures** | 6 architectures (CNN, ResNet-18, VGG-16, VAE, TinyBERT, DDPM) |
| **Maintainers** | Mahdieh Pouresmaeil, Reda Bellafqira |
| **Contact Email** | mahdieh.pouresmaeil@imt-atlantique.fr |
| **Status** | Active Development |
| **Last Updated** | February 25, 2026 |

---

## Overview

**WatDNN** is a comprehensive research framework investigating **Knowledge Distillation as a Defense Mechanism Against Obfuscation Attacks** on watermarked Deep Neural Networks. This project explores how knowledge distillation can be leveraged to **counteract obfuscation attacks** that attempt to obscure or remove watermarks from neural network models.

### Core Research Focus

This research investigates a paradigm shift: rather than using distillation to remove watermarks (as an attack) or model compression, this project examines how **distillation can be used as a defensive strategy** to preserve and protect watermarks against obfuscation attacks. 

### Problem Statement

Obfuscation attacks attempt to obscure watermark signatures in neural networks by changing the topology of the network, making them impossible to detect and verify in white-box watermarking schemes. White-box watermarking schemes can be vulnerable to sophisticated obfuscation techniques. This research proposes and evaluates **knowledge distillation as a defensive countermeasure** that tries to preserve watermark information by model distillation, when the original model has been obfuscated




## Project Structure

This project is organized into several key components for evaluating distillation-based defenses against obfuscation:

- **`attacks/`** - Contains obfuscation attack implementations and defense mechanisms:
  - `distillation.py` - **DEFENSE**: Knowledge distillation strategies to preserve and protect watermarks against obfuscation
  - `pruning.py` - an additional attack (removes unnecessary parameters to obscure watermarks)
  - `dummy_neurons.py` - Obfuscation attack  (obscures watermark by changing the topology of the network)
  - `ambiguity.py` -  an additional attack(creates confusion in watermark detection)
  
- **`models/`** - Pre-built model architectures for watermark evaluation

- **`networks/`** - Neural network implementations (MLP, CNN, ResNet, VGG, VAE, etc.)

- **`configs/`** - Configuration files for:
  - Data loading and preprocessing
  - Model training parameters
  - Watermark embedding strategies
  - Distillation-based defense settings
  - Obfuscation attack configurations

- **`datasets/`** - Prepared dataset loaders and utilities

- **`DDPM/`** - Diffusion models for adversarial data generation, distillation defense, and obfuscation attack scenarios

- **`VAE/`** - Variational Autoencoders for latent space exploration and synthetic data generation, and the test for distillation defense against obfuscation attacks

- **`watermark/`** - Watermarking scheme implementations (10 different schemes)

- **`results/`** - Output directory containing:
  - Trained base models
  - Watermarked models
  - Defense evaluation results
  - Statistical analyses

- **`sentiments/`** - implementation of distillation-based defense against obfuscation attacks on transfomer_based model (TinyBERT) on SST-2 dataset

- **`tests/`** - Testing utilities and evaluation metrics

- **`util/`** - Helper functions for metrics, and common operations

The project is developed in Python and provides a comprehensive framework for:
1. **Watermark Embedding** - Apply various watermarking schemes to neural networks
2. **Obfuscation Attacks** - Test how attacks attempt to obscure watermarks
3. **Distillation-based Defense** - Apply knowledge distillation to counteract obfuscation
4. **Robustness Evaluation** - Measure watermark persistence and detectability after obfuscation and defense



### Supported Datasets

We have configured all watermarking schemes and removal attacks for multiple image classification datasets:

- **CIFAR-10** (32×32 pixels, 10 classes)
- **CIFAR-100** (32×32 pixels, 100 classes)
- **MNIST** (28×28 pixels, 10 classes)
- **CelebA** (218×178 pixels, facial attributes)
- **SST-2** (Stanford Sentiment Treebank, for sentiment analysis with TinyBERT)

### Supported Model Architectures


- **CNN** - Convolutional Neural Network
- **ResNet-18** - Residual Network
- **VGG-16** - Visual Geometry Group
- **VAE** - Variational Autoencoder
- **TinyBERT** - Transformer-based model for sentiment analysis
- **DDPM** - Diffusion models for data generation in defense and attack scenarios

### Implemented Watermarking Schemes

The following watermarking schemes are implemented and can be combined with various models and datasets:

1. **DICTION** - DynamIC robusT whIte bOx watermarkiNg scheme (with adversarial learning) <http://arxiv.org/abs/2210.15745>
2. **DeepSigns** - Dynamic white-box watermarking with feature map preservation <https://arxiv.org/abs/1804.00750>
3. **Uchida** - Bit-trigger based watermarking approach <https://arxiv.org/abs/1701.04082>
4. **Encryption Resistant Scheme (RES_ENCRYPT)** - survive or resist encryption attacks <https://ieeexplore.ieee.org/document/9746461>
5. **RIGA** - Regularization-based watermarking scheme <https://dl.acm.org/doi/10.1145/3442381.3450000>
6. **PASSPORT** - Passport-protected networks for IP protection <https://arxiv.org/abs/1909.07830>
7. **HUFU** - Robust watermarking with feature-level protection. <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10038500> 
8. **STDM** - Spread Transform Dither Modulation <https://arxiv.org/abs/2012.14171>
9. **GREEDY** - Greedy optimization-based watermarking <https://proceedings.mlr.press/v139/liu21x/liu21x.pdf>
10. **LAPLACIAN** - Laplacian-based watermark distribution <https://arxiv.org/abs/2208.10973>



### Implemented Attack Methods (Obfuscation Techniques)

This framework implements  **obfuscation attack strategies** that attempt to obscure and remove watermarks by topology change. The primary contribution is evaluating **distillation-based defense mechanisms** against  obfuscation attacks:
additionally other types of attacks are implemented to evaluate the robustness of the watermarking schemes against attacks.


1. **Dummy Neurons Attack**
   - Insertion of dummy neurons to obscure the watermark.
   - Attempts to confuse watermark verification by adding irrelevant computations
   - **Defense**: Distillation filters out irrelevant neurons through knowledge transfer
   - Tests whether watermark robustness survives obfuscation


2. **Pruning Attack**
   - Removes parameters to obscure watermark information
   - Tests the resilience of watermarks to model compression
   - Evaluates watermark recovery after parameter removal

3. **Fine-Tuning Attack**
   - Continued training distorts watermark embeddings
   - Evaluates watermark stability under model drift

4. **Ambiguity Attack**
   - Creates ambiguity in the model watermark to create doubt in watermark verification
   - Tests watermark clarity preservation

5. **Overwriting Attack**
   - Attempts to embed conflicting watermarks
   - Tests the uniqueness and separation of watermark spaces
   - Evaluates watermark interference resilience


### Operations Supported

- **TRAIN** - Train a base model from scratch
- **WATERMARKING** - Embed a watermark into a model
- **TRAIN-WATERMARKING** - Train and watermark in one step
- **DISTILLATION** - Perform knowledge distillation 
- **PRUNING** - Apply pruning to the model
- **FINE_TUNING** - Fine-tune the model
- **OVERWRITING** - Attempt to overwrite watermarks
- **DUMMY_NEURONS** - Insert dummy neurons (obfuscation attack)
- **AMBIGUITY** - Apply ambiguity attacks


## Requirements

To set up the project environment, you'll need:

- **Python** >= 3.10
- **PyTorch** >= 2.2.2
- **NumPy** >= 1.23.5
- **SciPy** >= 1.10.0
- **Matplotlib** >= 3.5.3
- **torchvision** >= 0.17.2
- **tqdm** >= 4.65.0
- **Pillow** >= 10.3.0
- **setuptools**

For the complete list of dependencies, see `requirements.txt`.

## Setup

1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running Tests

The project provides a flexible testing framework to evaluate different combinations of watermarking schemes, models, and attacks.

### Quick Start

Execute the `run_tests.sh` script to perform various tests:

```bash
bash ./run_tests.sh
```

This script can be customized to run specific combinations of methods, models, and operations by modifying the parameters within the script.
- Runs the `test_case.py` script with different combinations of methods, models, and operations
- Outputs results to organized directories under `outs/.`


### Results and Outputs

Results are organized in the `outs/` directory by operation type:

```
outs/
├── TRAIN/
├── WATERMARKING/
├── DISTILLATION/
├── PRUNING/
├── FINE_TUNING/
├── OVERWRITING/
├── DUMMY_NEURONS/
├── AMBIGUITY/
```

Each directory contains logs and model files for the corresponding operation.

## Configuration Files

This project includes several configuration files that allow you to customize various aspects of the data, training, and watermarking processes.

- `cf_data.py`: This file contains configurations related to data, such as batch size and the path to save it. For example:

    ```python
    config_data = {
        "database": "my_database", # CIFAR-10 or MNIST
        "batch_size": 128, # batch size
        "device": "cuda", # device
        "path_to_save_data": "/data/my_data" # path to save data 
    }
    ```

- `cf_train` directory: This directory contains configuration files for training different models. For example, `cf_mlp.py` includes all the hyperparameters for training the MLP model and the path to save the trained model. For example:

    ```python
    config_model = {
        "lr": 0.001,  # learning rate
        "epochs": 50, # number of epochs
        "wd": 0, # weight decay
        "opt": "Adam", # optimizer
        "batch_size": config_data["batch_size"], # batch size
        "architecture": "MLP", # model architecture
        "milestones": [25, 45], # milestones
        "gamma": 0.1, # gamma related to the scheduler
        "criterion": nn.CrossEntropyLoss(), # loss function
        "scheduler": "MultiStepLR", # scheduler
        "device": config_data["device"], # device
        "database": config_data["database"], # database
        "momentum": 0, # momentum for the optimizer 
        "save_path": "/models/my_model" # path to save the model
    }
    ```

- `cf_watermark` directory: This directory contains configuration files for different watermarking schemes. For example, `cf_diction.py` includes the watermarking parameters for each model to watermark, such as the size of the watermark, the layer to watermark, the number of epochs to embed the watermark, and so on. For example:

    ```python
    cf_mlp_embed = {
        "configuration": config_model, # model configuration
        "database": config_model["database"], # database
        "watermark_size": 512, # watermark size
        "epochs": 10000, # the maximum number of epochs to embed the watermark
        "epoch_check": 30, # the number of epochs to check the watermark, if ber is 0, the watermark is embedded, and the process stops
        "batch_size": int(config_model["batch_size"] * 0.5), #  batch size of the trigger set, e.g. half size of the batch size of the training set    
        "mean": 0, # mean to generate the trigger set 
        "std": 1, # std to generate the trigger set
        # To customize the trigger set, you can add a square into the images of the trigger set, the following parameters are used to generate the square
        "square_size": 5, # square size to add it into the trigger set
        "start_x": 0, # start x to generate the square     
        "start_y": 0, # start y to generate the square
        "square_value": 1, # value of the square
    
        "n_features": 0.5, #  the percentage of features to be used in the watermarking process
        "lambda": 1e-0, # the regularization parameter to train the projection model
        "layer_name": "fc2",  # the layer name to be watermarked
        "lr": 1e-3, # learning rate to embed the watermark into the target model
        "wd": 0, # weight decay to embed the watermark into the target model
        "opt": "Adam", # optimizer to embed the watermark into the target model
        "scheduler": "MultiStepLR", # scheduler to embed the watermark into the target model
        "architecture": config_model['architecture'], # architecture of the model to watermark
        "momentum": 0, # momentum to embed the watermark into the target model
        "milestones": [20, 3000], # milestones to embed the watermark into the target model
        "gamma": .1,    # gamma to embed the watermark into the target model
        "criterion": config_model["criterion"], # loss function to embed the watermark into the target model
        "device": config_model["device"],   # device to embed the watermark into the target model
        "path_model": config_model["save_path"], # path to load the original model
        "save_path": "results/watermarked_models/diction/" + config_model['architecture'].lower() + "/_lfc2_wat512_ep10000_epc30.pth" # path to save the watermarked model
    }
    ```




### 1. Distillation-Based Defense (DEFENSE MECHANISM)

**The Primary Contribution: Using Knowledge Distillation as a Defense**

Distillation is not an attack in this framework—it is the **primary defense mechanism** against obfuscation attacks. Instead of a student model learning from a watermarked teacher to steal the watermark, here we use distillation to **preserve and recover watermarks** that have been obfuscated.

**How Defense Works:**
- When a watermarked model (white-box watermarking) is subjected to obfuscation attacks, the watermark becomes impossible to verify 
- Apply knowledge distillation to transfer the essential properties (including watermark) to a new model
- The distillation process acts as a filter, removing noise and obfuscation while preserving critical watermark information
- Feature-level and output-level alignment ensures watermark robustness is maintained


### 2. Dummy Neurons Attack (Obfuscation via topology change)

Inserts irrelevant neurons to obscure the watermark.

**How Obfuscation Works:**
- Add dummy neurons that perform no meaningful effect on the model's functionality. 
- Increases model complexity without improving accuracy
- destroy watermark verification in white-box watermarking schemes


**Defense Strategy:**
- Student model learns only essential features and watermark patterns
- Results in a cleaner, smaller model with preserved watermark



## Jupyter Notebooks

Several analysis notebooks are provided in `sentiments/` and `DDPM/`:

- **Sentiment**: Analyze transformer-based model tinyBERT on SST-2 dataset for distillation-based defense against obfuscation attacks
- **DDPM Models**: Diffusion model implementations on Cifar10 dataset for adversarial data generation in distillation defense and obfuscation attack scenarios
- **VAE Models**: Variational autoencoders for latent space exploration on the celebA dataset  for testing distillation defense against obfuscation attacks


## Contributing

Contributions are welcome. Please submit a pull request with your changes. If you have any questions, please contact me at mahdieh.pouresmaeil@imt-atlantique.fr

## License

This project is licensed under the terms of the MIT license.
