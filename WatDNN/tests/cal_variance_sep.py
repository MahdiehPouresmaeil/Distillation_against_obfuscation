import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/home/latim/PycharmProjects/WatDNN")

from networks.cnn import CnnModel
import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse


def test_laplacian_distribution(data, alpha=0.05):
    """
    Test if data follows a Laplacian distribution using multiple methods.
    The Laplacian parameters (location and scale) are always estimated from the data.
    Note: The Kolmogorov-Smirnov test p-value may be optimistic because parameters are estimated from the data.

    Args:
        data: 1D array or tensor of data
        alpha: Significance level for statistical tests

    Returns:
        dict: Results of various tests
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy().flatten()
    else:
        data = np.array(data).flatten()

    # Remove any NaN or infinite values
    data = data[np.isfinite(data)]

    if len(data) < 10:
        return {"error": "Not enough data points for testing"}

    results = {}

    # 1. Kolmogorov-Smirnov test
    try:
        # Fit Laplacian parameters (location=median, scale=mean absolute deviation)
        loc_est = np.median(data)  # More robust than mean
        scale_est = np.mean(np.abs(data - loc_est))  # Mean absolute deviation

        # KS test against fitted Laplacian
        ks_stat, ks_pvalue = stats.kstest(data, lambda x: stats.laplace.cdf(x, loc=loc_est, scale=scale_est))
        results['ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'is_laplacian': ks_pvalue > alpha,
            'fitted_loc': loc_est,
            'fitted_scale': scale_est,
            'note': 'Parameters estimated from data; p-value may be optimistic.'
        }
    except Exception as e:
        results['ks_test'] = {'error': str(e)}

    # 2. Anderson-Darling test (if available)
    try:
        # Transform to standard Laplacian
        standardized = (data - loc_est) / scale_est
        ad_stat, ad_critical_values, ad_significance_levels = stats.anderson(standardized,
                                                                             dist='logistic')  # Closest available
        results['anderson_test'] = {
            'statistic': ad_stat,
            'critical_values': ad_critical_values,
            'significance_levels': ad_significance_levels
        }
    except Exception as e:
        results['anderson_test'] = {'error': str(e)}

    # 3. Kurtosis test (Laplacian has kurtosis = 6, excess kurtosis = 3)
    try:
        kurt_stat = stats.kurtosis(data, fisher=True)  # Excess kurtosis
        # Test if kurtosis is significantly different from 3 (Laplacian excess kurtosis)
        n = len(data)
        se_kurt = np.sqrt(24 / n)  # Standard error of excess kurtosis
        z_score = (kurt_stat - 3) / se_kurt
        kurt_pvalue = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

        results['kurtosis_test'] = {
            'excess_kurtosis': kurt_stat,
            'expected_laplacian': 3.0,
            'z_score': z_score,
            'p_value': kurt_pvalue,
            'is_laplacian': kurt_pvalue > alpha
        }
    except Exception as e:
        results['kurtosis_test'] = {'error': str(e)}

    # 4. Skewness test (Laplacian should have skewness = 0)
    try:
        skew_stat = stats.skew(data)
        skew_test_stat, skew_pvalue = stats.skewtest(data)
        results['skewness_test'] = {
            'skewness': skew_stat,
            'expected_laplacian': 0.0,
            'test_statistic': skew_test_stat,
            'p_value': skew_pvalue,
            'is_symmetric': skew_pvalue > alpha
        }
    except Exception as e:
        results['skewness_test'] = {'error': str(e)}

    # 5. Overall assessment
    laplacian_indicators = []
    if 'ks_test' in results and 'is_laplacian' in results['ks_test']:
        laplacian_indicators.append(results['ks_test']['is_laplacian'])
    if 'kurtosis_test' in results and 'is_laplacian' in results['kurtosis_test']:
        laplacian_indicators.append(results['kurtosis_test']['is_laplacian'])
    if 'skewness_test' in results and 'is_symmetric' in results['skewness_test']:
        laplacian_indicators.append(results['skewness_test']['is_symmetric'])

    if laplacian_indicators:
        confidence_score = sum(laplacian_indicators) / len(laplacian_indicators)
        results['overall'] = {
            'likely_laplacian': confidence_score > 0.5,
            'confidence_score': confidence_score,
            'tests_passed': sum(laplacian_indicators),
            'total_tests': len(laplacian_indicators)
        }

    return results


def show_model_weights_with_laplacian_test(model, alpha=0.15, plot_histograms=False):
    """
    Enhanced function to display model weights with Laplacian distribution testing.

    Args:
        model: PyTorch model
        alpha: Significance level for statistical tests
        plot_histograms: Whether to plot histograms for each layer
    """
    print("=" * 80)
    print("MODEL WEIGHTS ANALYSIS WITH LAPLACIAN DISTRIBUTION TESTING")
    print("=" * 80)

    total_layers = 0
    laplacian_layers = 0

    for name, param in model.named_parameters():
        total_layers += 1

        print(f"\nLayer: {name}")
        print(f"Shape: {param.shape}")
        print(f"Type: {param.dtype}")
        print(f"Requires grad: {param.requires_grad}")
        print(f"Device: {param.device}")
        print(f"Total parameters: {param.numel():,}")

        # Basic statistics
        with torch.no_grad():
            weights = param.flatten()
            mean_val = weights.mean().item()
            var_val = weights.var().item()
            std_val = weights.std().item()

            print(f"\nBasic Statistics:")
            print(f"  Mean: {mean_val:.6f}")
            print(f"  Variance: {var_val:.6f}")
            print(f"  Std: {std_val:.6f}")
            print(f"  Min: {weights.min().item():.6f}")
            print(f"  Max: {weights.max().item():.6f}")
            print(f"  Median: {weights.median().item():.6f}")

            # Additional statistics for Laplacian assessment
            mad = torch.mean(torch.abs(weights - weights.median())).item()  # Mean Absolute Deviation
            iqr = (torch.quantile(weights, 0.75) - torch.quantile(weights, 0.25)).item()  # Interquartile Range

            print(f"  Mean Abs Dev: {mad:.6f}")
            print(f"  IQR: {iqr:.6f}")

        # Test for Laplacian distribution
        print(f"\nLaplacian Distribution Testing (α={alpha}):")
        laplacian_results = test_laplacian_distribution(weights, alpha=alpha)

        if 'error' in laplacian_results:
            print(f"  ❌ Error: {laplacian_results['error']}")
        else:
            # Display test results
            if 'ks_test' in laplacian_results:
                ks = laplacian_results['ks_test']
                if 'error' not in ks:
                    print(f"  📊 Kolmogorov-Smirnov Test:")
                    print(f"     Statistic: {ks['statistic']:.6f}")
                    print(f"     P-value: {ks['p_value']:.6f}")
                    print(f"     Result: {'✅ Laplacian' if ks['is_laplacian'] else '❌ Not Laplacian'}")
                    print(f"     Fitted location: {ks['fitted_loc']:.6f}")
                    print(f"     Fitted scale: {ks['fitted_scale']:.6f}")
                    print(f"     Note: {ks['note']}")

            if 'kurtosis_test' in laplacian_results:
                kurt = laplacian_results['kurtosis_test']
                if 'error' not in kurt:
                    print(f"  📊 Kurtosis Test:")
                    print(f"     Excess kurtosis: {kurt['excess_kurtosis']:.3f} (Laplacian: 3.0)")
                    print(f"     P-value: {kurt['p_value']:.6f}")
                    print(f"     Result: {'✅ Compatible' if kurt['is_laplacian'] else '❌ Incompatible'}")

            if 'skewness_test' in laplacian_results:
                skew = laplacian_results['skewness_test']
                if 'error' not in skew:
                    print(f"  📊 Skewness Test:")
                    print(f"     Skewness: {skew['skewness']:.3f} (Laplacian: 0.0)")
                    print(f"     P-value: {skew['p_value']:.6f}")
                    print(f"     Result: {'✅ Symmetric' if skew['is_symmetric'] else '❌ Asymmetric'}")

            # Overall assessment
            if 'overall' in laplacian_results:
                overall = laplacian_results['overall']
                print(f"  🎯 Overall Assessment:")
                print(f"     Likely Laplacian: {'✅ YES' if overall['likely_laplacian'] else '❌ NO'}")
                print(f"     Confidence: {overall['confidence_score']:.1%}")
                print(f"     Tests passed: {overall['tests_passed']}/{overall['total_tests']}")

                if overall['likely_laplacian']:
                    laplacian_layers += 1

        # Optional: Plot histogram
        if plot_histograms:
            try:
                plt.figure(figsize=(10, 4))

                # Subplot 1: Histogram
                plt.subplot(1, 2, 1)
                weights_np = weights.detach().cpu().numpy()
                plt.hist(weights_np, bins=50, density=True, alpha=0.7, label='Actual')

                # Overlay fitted Laplacian if available
                if 'ks_test' in laplacian_results and 'fitted_loc' in laplacian_results['ks_test']:
                    ks = laplacian_results['ks_test']
                    x_range = np.linspace(weights_np.min(), weights_np.max(), 100)
                    fitted_laplacian = stats.laplace.pdf(x_range,
                                                         loc=ks['fitted_loc'],
                                                         scale=ks['fitted_scale'])
                    plt.plot(x_range, fitted_laplacian, 'r--', label='Fitted Laplacian')

                plt.title(f'{name}\nWeight Distribution')
                plt.xlabel('Weight Value')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Subplot 2: Q-Q plot against Laplacian
                plt.subplot(1, 2, 2)
                if 'ks_test' in laplacian_results and 'fitted_loc' in laplacian_results['ks_test']:
                    ks = laplacian_results['ks_test']
                    stats.probplot(weights_np, dist=stats.laplace,
                                   sparams=(ks['fitted_loc'], ks['fitted_scale']),
                                   plot=plt)
                    plt.title('Q-Q Plot vs Laplacian')
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"  Warning: Could not plot histogram - {e}")

        print("=" * 60)

    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"Total layers analyzed: {total_layers}")
    print(f"Layers likely following Laplacian distribution: {laplacian_layers}")
    print(f"Percentage: {100 * laplacian_layers / total_layers:.1f}%")


def quick_laplacian_check(model, layer_name_filter=None):
    """
    Quick check for Laplacian distribution in specific layers.

    Args:
        model: PyTorch model
        layer_name_filter: String to filter layer names (e.g., 'conv', 'weight')
    """
    print("🔍 QUICK LAPLACIAN CHECK")
    print("=" * 40)

    for name, param in model.named_parameters():
        if layer_name_filter is None or layer_name_filter.lower() in name.lower():
            weights = param.flatten()
            results = test_laplacian_distribution(weights)

            if 'overall' in results:
                status = "✅" if results['overall']['likely_laplacian'] else "❌"
                confidence = results['overall']['confidence_score']
                print(f"{status} {name}: {confidence:.1%} confidence")
            else:
                print(f"❓ {name}: Could not test")


def show_model_weights_basic(model):
    """
    Basic function to display model weights.

    Args:
        model: PyTorch model
    """
    print("=" * 60)
    print("MODEL WEIGHTS OVERVIEW")
    print("=" * 60)

    for name, param in model.named_parameters():
        print(f"\nLayer: {name}")
        print(f"Shape: {param.shape}")
        print(f"Type: {param.dtype}")
        print(f"Requires grad: {param.requires_grad}")
        print(f"Device: {param.device}")
        print(f"Min: {param.min().item():.6f}")
        print(f"Max: {param.max().item():.6f}")
        print(f"Mean: {param.mean().item():.6f}")
        print(f"Std: {param.std().item():.6f}")
        print("-" * 40)


def calculate_variance(model_path, alpha=0.5):
    # Load the saved file
    checkpoint = torch.load(model_path, weights_only=False)

    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Model type: {type(checkpoint['model'])}")

    # The 'model' key contains the actual model object, not state_dict
    init_model = checkpoint['model']
    show_model_weights_with_laplacian_test(init_model, alpha=alpha)
    print("\n🎯 WATERMARKING TARGET ANALYSIS:")
    quick_laplacian_check(init_model, layer_name_filter='conv')
    quick_laplacian_check(init_model, layer_name_filter='fc')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Analyze model weights for Laplacian distribution.")
    # parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint (.pth)')
    # parser.add_argument('--alpha', type=float, default=0.5, help='Significance level for Laplacian tests (default: 0.5)')
    # args = parser.parse_args()

    model_path = "/home/latim/PycharmProjects/WatDNN/results/trained_models/resnet18/_dbcifar10_ep200_bs128.pth"
    # calculate_variance(args.model, alpha=args.alpha)
    calculate_variance(model_path, alpha=0.5)





"""
MODEL WEIGHTS ANALYSIS WITH LAPLACIAN DISTRIBUTION TESTING
================================================================================

Layer: conv1.weight
Shape: torch.Size([32, 3, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 864

Basic Statistics:
  Mean: -0.001233
  Variance: 0.025277
  Std: 0.158988
  Min: -0.399560
  Max: 0.420664
  Median: 0.001005
  Mean Abs Dev: 0.132127
  IQR: 0.245007

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.062685
     P-value: 0.002149
     Result: ❌ Not Laplacian
     Fitted location: 0.001100
     Fitted scale: 0.132127
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.619 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.015 (Laplacian: 0.0)
     P-value: 0.854801
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: conv1.bias
Shape: torch.Size([32])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 32

Basic Statistics:
  Mean: -0.072309
  Variance: 0.091188
  Std: 0.301973
  Min: -0.782645
  Max: 0.288034
  Median: -0.013004
  Mean Abs Dev: 0.233937
  IQR: 0.386430

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.150568
     P-value: 0.421243
     Result: ✅ Laplacian
     Fitted location: 0.007265
     Fitted scale: 0.233937
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.179 (Laplacian: 3.0)
     P-value: 0.000242
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.918 (Laplacian: 0.0)
     P-value: 0.024830
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: conv2.weight
Shape: torch.Size([32, 32, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 9,216

Basic Statistics:
  Mean: -0.013637
  Variance: 0.012853
  Std: 0.113373
  Min: -0.569453
  Max: 0.427205
  Median: -0.004498
  Mean Abs Dev: 0.087110
  IQR: 0.141752

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.047756
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.004486
     Fitted scale: 0.087110
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.706 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.318 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: conv2.bias
Shape: torch.Size([32])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 32

Basic Statistics:
  Mean: 0.030613
  Variance: 0.011578
  Std: 0.107602
  Min: -0.241345
  Max: 0.188014
  Median: 0.044976
  Mean Abs Dev: 0.080782
  IQR: 0.119930

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.096869
     P-value: 0.896672
     Result: ✅ Laplacian
     Fitted location: 0.047380
     Fitted scale: 0.080782
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.499 (Laplacian: 3.0)
     P-value: 0.003882
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.863 (Laplacian: 0.0)
     P-value: 0.033286
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: conv3.weight
Shape: torch.Size([64, 32, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 18,432

Basic Statistics:
  Mean: -0.005303
  Variance: 0.010682
  Std: 0.103355
  Min: -0.947209
  Max: 0.410471
  Median: -0.000000
  Mean Abs Dev: 0.072462
  IQR: 0.105585

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.054039
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.072462
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.514 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.626 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: conv3.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.092095
  Variance: 0.037945
  Std: 0.194796
  Min: -0.425404
  Max: 0.643045
  Median: 0.044311
  Mean Abs Dev: 0.142233
  IQR: 0.215177

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.144914
     P-value: 0.123024
     Result: ✅ Laplacian
     Fitted location: 0.046581
     Fitted scale: 0.142233
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.792 (Laplacian: 3.0)
     P-value: 0.000311
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.565 (Laplacian: 0.0)
     P-value: 0.055175
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ✅ YES
     Confidence: 66.7%
     Tests passed: 2/3
============================================================

Layer: conv4.weight
Shape: torch.Size([64, 64, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 36,864

Basic Statistics:
  Mean: -0.004766
  Variance: 0.006255
  Std: 0.079089
  Min: -0.435153
  Max: 0.602465
  Median: -0.000077
  Mean Abs Dev: 0.050306
  IQR: 0.049732

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.117845
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000077
     Fitted scale: 0.050306
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.949 (Laplacian: 3.0)
     P-value: 0.047460
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.085 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: conv4.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.117242
  Variance: 0.014200
  Std: 0.119165
  Min: -0.067583
  Max: 0.447432
  Median: 0.095827
  Mean Abs Dev: 0.104262
  IQR: 0.203142

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.169356
     P-value: 0.044932
     Result: ❌ Not Laplacian
     Fitted location: 0.108851
     Fitted scale: 0.104262
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.721 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.474 (Laplacian: 0.0)
     P-value: 0.102976
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: fc1.weight
Shape: torch.Size([512, 1600])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 819,200

Basic Statistics:
  Mean: -0.002433
  Variance: 0.002144
  Std: 0.046308
  Min: -0.357200
  Max: 0.383222
  Median: -0.000122
  Mean Abs Dev: 0.030963
  IQR: 0.038470

Laplacian Distribution Testing (α=0.05):
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2280: RuntimeWarning: overflow encountered in exp
  tmp2 = exp(tmp)
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2282: RuntimeWarning: invalid value encountered in divide
  np.sum(tmp*(1.0-tmp2)/(1+tmp2), axis=0) + N]
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2286: RuntimeWarning: The iteration is not making good progress, as measured by the 
 improvement from the last ten iterations.
  sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-5)
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.107738
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000122
     Fitted scale: 0.030963
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.102 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.066 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: fc1.bias
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: -0.092891
  Variance: 0.001179
  Std: 0.034332
  Min: -0.189753
  Max: 0.057758
  Median: -0.093609
  Mean Abs Dev: 0.026203
  IQR: 0.042032

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.045400
     P-value: 0.234700
     Result: ✅ Laplacian
     Fitted location: -0.093432
     Fitted scale: 0.026203
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 1.591 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.444 (Laplacian: 0.0)
     P-value: 0.000068
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: fc2.weight
Shape: torch.Size([10, 512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 5,120

Basic Statistics:
  Mean: -0.025557
  Variance: 0.022544
  Std: 0.150146
  Min: -0.476325
  Max: 0.607945
  Median: -0.034858
  Mean Abs Dev: 0.120453
  IQR: 0.207892

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.060682
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.034833
     Fitted scale: 0.120453
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.092 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.333 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: fc2.bias
Shape: torch.Size([10])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 10

Basic Statistics:
  Mean: 0.008118
  Variance: 0.012712
  Std: 0.112748
  Min: -0.159060
  Max: 0.212758
  Median: -0.009479
  Mean Abs Dev: 0.075564
  IQR: 0.062869

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.176562
     P-value: 0.862622
     Result: ✅ Laplacian
     Fitted location: -0.008629
     Fitted scale: 0.075564
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.410 (Laplacian: 3.0)
     P-value: 0.027712
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.471 (Laplacian: 0.0)
     P-value: 0.403626
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ✅ YES
     Confidence: 66.7%
     Tests passed: 2/3
============================================================

Layer: bt_n.weight
Shape: torch.Size([1600])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 1,600

Basic Statistics:
  Mean: 0.332777
  Variance: 0.061836
  Std: 0.248669
  Min: -0.496517
  Max: 0.937402
  Median: 0.415273
  Mean Abs Dev: 0.203592
  IQR: 0.514014

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.198584
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.415309
     Fitted scale: 0.203592
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -1.047 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.341 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: bt_n.bias
Shape: torch.Size([1600])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 1,600

Basic Statistics:
  Mean: 0.031905
  Variance: 0.006996
  Std: 0.083641
  Min: -1.255419
  Max: 1.184805
  Median: 0.020988
  Mean Abs Dev: 0.038366
  IQR: 0.056457

Laplacian Distribution Testing (α=0.05):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.193215
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.021060
     Fitted scale: 0.038366
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 102.412 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -1.354 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

📋 SUMMARY:
Total layers analyzed: 14
Layers likely following Laplacian distribution: 2
Percentage: 14.3%

🎯 WATERMARKING TARGET ANALYSIS:
🔍 QUICK LAPLACIAN CHECK
========================================
❌ conv1.weight: 33.3% confidence
❌ conv1.bias: 33.3% confidence
❌ conv2.weight: 0.0% confidence
❌ conv2.bias: 33.3% confidence
❌ conv3.weight: 0.0% confidence
✅ conv3.bias: 66.7% confidence
❌ conv4.weight: 0.0% confidence
❌ conv4.bias: 33.3% confidence
🔍 QUICK LAPLACIAN CHECK
========================================
❌ fc1.weight: 0.0% confidence
❌ fc1.bias: 33.3% confidence
❌ fc2.weight: 0.0% confidence
✅ fc2.bias: 66.7% confidence

*********************************************************************************************************************
***********************************************************************************************************************
Checkpoint keys: ['model', 'acc', 'epoch', 'model_state_dict']
Model type: <class 'networks.resnet18_two_linear.ResNet18TwoLinear'>
================================================================================
MODEL WEIGHTS ANALYSIS WITH LAPLACIAN DISTRIBUTION TESTING
================================================================================

Layer: base.conv1.weight
Shape: torch.Size([64, 3, 7, 7])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 9,408

Basic Statistics:
  Mean: -0.000360
  Variance: 0.025945
  Std: 0.161076
  Min: -0.980172
  Max: 0.912772
  Median: 0.000000
  Mean Abs Dev: 0.099508
  IQR: 0.111106

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.110792
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.000000
     Fitted scale: 0.099508
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 4.461 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.112 (Laplacian: 0.0)
     P-value: 0.000010
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.bn1.weight
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.282717
  Variance: 0.030446
  Std: 0.174487
  Min: -0.003232
  Max: 0.717948
  Median: 0.311939
  Mean Abs Dev: 0.127347
  IQR: 0.146920

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.177186
     P-value: 0.031468
     Result: ❌ Not Laplacian
     Fitted location: 0.316815
     Fitted scale: 0.127347
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.300 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.337 (Laplacian: 0.0)
     P-value: 0.238020
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.bn1.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: -0.003899
  Variance: 0.039544
  Std: 0.198857
  Min: -0.708473
  Max: 0.360830
  Median: -0.000000
  Mean Abs Dev: 0.105887
  IQR: 0.066064

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.218966
     P-value: 0.003563
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.105887
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 3.981 (Laplacian: 3.0)
     P-value: 0.109323
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -1.622 (Laplacian: 0.0)
     P-value: 0.000009
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer1.0.conv1.weight
Shape: torch.Size([64, 64, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 36,864

Basic Statistics:
  Mean: -0.004383
  Variance: 0.007783
  Std: 0.088219
  Min: -1.481887
  Max: 0.664186
  Median: -0.000000
  Mean Abs Dev: 0.053405
  IQR: 0.063989

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.128301
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.053405
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 13.419 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -1.481 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer1.0.bn1.weight
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.349438
  Variance: 0.017000
  Std: 0.130386
  Min: 0.000000
  Max: 0.588041
  Median: 0.340481
  Mean Abs Dev: 0.096040
  IQR: 0.134236

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.115007
     P-value: 0.339429
     Result: ❌ Not Laplacian
     Fitted location: 0.340771
     Fitted scale: 0.096040
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.991 (Laplacian: 3.0)
     P-value: 0.001038
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.767 (Laplacian: 0.0)
     P-value: 0.012135
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer1.0.bn1.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: -0.116295
  Variance: 0.036200
  Std: 0.190263
  Min: -0.539920
  Max: 0.263482
  Median: -0.111790
  Mean Abs Dev: 0.157052
  IQR: 0.286156

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.110866
     P-value: 0.382946
     Result: ❌ Not Laplacian
     Fitted location: -0.104885
     Fitted scale: 0.157052
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.655 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.033 (Laplacian: 0.0)
     P-value: 0.905479
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer1.0.conv2.weight
Shape: torch.Size([64, 64, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 36,864

Basic Statistics:
  Mean: -0.004185
  Variance: 0.007307
  Std: 0.085481
  Min: -0.751847
  Max: 0.639801
  Median: -0.000358
  Mean Abs Dev: 0.059958
  IQR: 0.085413

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.052118
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000357
     Fitted scale: 0.059958
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 3.286 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.002 (Laplacian: 0.0)
     P-value: 0.881611
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer1.0.bn2.weight
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.339734
  Variance: 0.033239
  Std: 0.182315
  Min: -0.007286
  Max: 0.658922
  Median: 0.360405
  Mean Abs Dev: 0.152766
  IQR: 0.287235

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.098932
     P-value: 0.525776
     Result: ✅ Laplacian
     Fitted location: 0.362276
     Fitted scale: 0.152766
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -1.006 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.271 (Laplacian: 0.0)
     P-value: 0.338450
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer1.0.bn2.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: -0.102756
  Variance: 0.038189
  Std: 0.195421
  Min: -0.485986
  Max: 0.652882
  Median: -0.132088
  Mean Abs Dev: 0.149359
  IQR: 0.223912

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.112072
     P-value: 0.369923
     Result: ❌ Not Laplacian
     Fitted location: -0.120384
     Fitted scale: 0.149359
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.226 (Laplacian: 3.0)
     P-value: 0.206189
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.873 (Laplacian: 0.0)
     P-value: 0.005156
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer1.1.conv1.weight
Shape: torch.Size([64, 64, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 36,864

Basic Statistics:
  Mean: -0.003778
  Variance: 0.008098
  Std: 0.089987
  Min: -0.940074
  Max: 0.663154
  Median: -0.003610
  Mean Abs Dev: 0.066928
  IQR: 0.104197

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.031813
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.003610
     Fitted scale: 0.066928
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 3.181 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.039 (Laplacian: 0.0)
     P-value: 0.002093
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer1.1.bn1.weight
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.318067
  Variance: 0.006739
  Std: 0.082090
  Min: 0.026951
  Max: 0.580272
  Median: 0.311899
  Mean Abs Dev: 0.059451
  IQR: 0.104259

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.085028
     P-value: 0.711610
     Result: ✅ Laplacian
     Fitted location: 0.311931
     Fitted scale: 0.059451
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.480 (Laplacian: 3.0)
     P-value: 0.395465
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.017 (Laplacian: 0.0)
     P-value: 0.950147
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ✅ YES
     Confidence: 66.7%
     Tests passed: 2/3
============================================================

Layer: base.layer1.1.bn1.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: -0.260360
  Variance: 0.022775
  Std: 0.150913
  Min: -0.599088
  Max: 0.067528
  Median: -0.260447
  Mean Abs Dev: 0.121490
  IQR: 0.190477

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.100645
     P-value: 0.503888
     Result: ✅ Laplacian
     Fitted location: -0.258537
     Fitted scale: 0.121490
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.270 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.009 (Laplacian: 0.0)
     P-value: 0.973558
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ✅ YES
     Confidence: 66.7%
     Tests passed: 2/3
============================================================

Layer: base.layer1.1.conv2.weight
Shape: torch.Size([64, 64, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 36,864

Basic Statistics:
  Mean: -0.006358
  Variance: 0.006395
  Std: 0.079967
  Min: -0.565140
  Max: 0.414076
  Median: -0.003251
  Mean Abs Dev: 0.058465
  IQR: 0.086933

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.027579
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.003250
     Fitted scale: 0.058465
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 1.720 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.076 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer1.1.bn2.weight
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: 0.298854
  Variance: 0.031543
  Std: 0.177603
  Min: -0.015995
  Max: 0.715038
  Median: 0.313095
  Mean Abs Dev: 0.146971
  IQR: 0.268943

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.114971
     P-value: 0.339793
     Result: ❌ Not Laplacian
     Fitted location: 0.319073
     Fitted scale: 0.146971
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.737 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.149 (Laplacian: 0.0)
     P-value: 0.595077
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer1.1.bn2.bias
Shape: torch.Size([64])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 64

Basic Statistics:
  Mean: -0.181932
  Variance: 0.037417
  Std: 0.193435
  Min: -0.789679
  Max: 0.194166
  Median: -0.153995
  Mean Abs Dev: 0.146623
  IQR: 0.219762

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.095519
     P-value: 0.570441
     Result: ✅ Laplacian
     Fitted location: -0.146416
     Fitted scale: 0.146623
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.803 (Laplacian: 3.0)
     P-value: 0.000333
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.787 (Laplacian: 0.0)
     P-value: 0.010314
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer2.0.conv1.weight
Shape: torch.Size([128, 64, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 73,728

Basic Statistics:
  Mean: -0.002356
  Variance: 0.006636
  Std: 0.081465
  Min: -0.483244
  Max: 0.686403
  Median: -0.005119
  Mean Abs Dev: 0.062488
  IQR: 0.099781

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.035937
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.005118
     Fitted scale: 0.062488
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 1.543 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.320 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.0.bn1.weight
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: 0.320407
  Variance: 0.001309
  Std: 0.036181
  Min: 0.223640
  Max: 0.405842
  Median: 0.319059
  Mean Abs Dev: 0.028439
  IQR: 0.047219

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.073187
     P-value: 0.477039
     Result: ❌ Not Laplacian
     Fitted location: 0.319277
     Fitted scale: 0.028439
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.092 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.001 (Laplacian: 0.0)
     P-value: 0.996591
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer2.0.bn1.bias
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: -0.214452
  Variance: 0.006411
  Std: 0.080066
  Min: -0.449452
  Max: 0.114539
  Median: -0.222947
  Mean Abs Dev: 0.057749
  IQR: 0.089152

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.054900
     P-value: 0.814688
     Result: ✅ Laplacian
     Fitted location: -0.222503
     Fitted scale: 0.057749
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.637 (Laplacian: 3.0)
     P-value: 0.402298
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.710 (Laplacian: 0.0)
     P-value: 0.001577
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer2.0.conv2.weight
Shape: torch.Size([128, 128, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 147,456

Basic Statistics:
  Mean: -0.002964
  Variance: 0.005258
  Std: 0.072512
  Min: -0.403413
  Max: 0.486881
  Median: -0.004063
  Mean Abs Dev: 0.055673
  IQR: 0.089521

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.033220
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.004063
     Fitted scale: 0.055673
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.967 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.238 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.0.bn2.weight
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: 0.312953
  Variance: 0.010871
  Std: 0.104263
  Min: -0.006094
  Max: 0.480884
  Median: 0.333609
  Mean Abs Dev: 0.079356
  IQR: 0.134380

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.078590
     P-value: 0.387834
     Result: ❌ Not Laplacian
     Fitted location: 0.334046
     Fitted scale: 0.079356
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.515 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.869 (Laplacian: 0.0)
     P-value: 0.000188
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.0.bn2.bias
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: -0.145016
  Variance: 0.004517
  Std: 0.067205
  Min: -0.305668
  Max: 0.004026
  Median: -0.146379
  Mean Abs Dev: 0.053749
  IQR: 0.092262

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.063174
     P-value: 0.662800
     Result: ✅ Laplacian
     Fitted location: -0.146339
     Fitted scale: 0.053749
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.217 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.014 (Laplacian: 0.0)
     P-value: 0.945926
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ✅ YES
     Confidence: 66.7%
     Tests passed: 2/3
============================================================

Layer: base.layer2.0.downsample.0.weight
Shape: torch.Size([128, 64, 1, 1])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 8,192

Basic Statistics:
  Mean: -0.008223
  Variance: 0.013205
  Std: 0.114912
  Min: -0.936710
  Max: 0.979517
  Median: -0.005413
  Mean Abs Dev: 0.081961
  IQR: 0.121610

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.023831
     P-value: 0.000179
     Result: ❌ Not Laplacian
     Fitted location: -0.005409
     Fitted scale: 0.081961
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 4.996 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.051 (Laplacian: 0.0)
     P-value: 0.058792
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.0.downsample.1.weight
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: 0.211364
  Variance: 0.012968
  Std: 0.113877
  Min: -0.023721
  Max: 0.445267
  Median: 0.198363
  Mean Abs Dev: 0.092368
  IQR: 0.152068

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.101876
     P-value: 0.130855
     Result: ❌ Not Laplacian
     Fitted location: 0.198958
     Fitted scale: 0.092368
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.622 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.051 (Laplacian: 0.0)
     P-value: 0.803735
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer2.0.downsample.1.bias
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: -0.145016
  Variance: 0.004517
  Std: 0.067205
  Min: -0.305668
  Max: 0.004026
  Median: -0.146379
  Mean Abs Dev: 0.053749
  IQR: 0.092262

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.063174
     P-value: 0.662800
     Result: ✅ Laplacian
     Fitted location: -0.146339
     Fitted scale: 0.053749
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.217 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.014 (Laplacian: 0.0)
     P-value: 0.945926
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ✅ YES
     Confidence: 66.7%
     Tests passed: 2/3
============================================================

Layer: base.layer2.1.conv1.weight
Shape: torch.Size([128, 128, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 147,456

Basic Statistics:
  Mean: -0.001839
  Variance: 0.005679
  Std: 0.075361
  Min: -0.361624
  Max: 0.541198
  Median: -0.004158
  Mean Abs Dev: 0.058276
  IQR: 0.094279

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.033288
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.004158
     Fitted scale: 0.058276
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.658 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.278 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.1.bn1.weight
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: 0.283323
  Variance: 0.001521
  Std: 0.038996
  Min: 0.012724
  Max: 0.374808
  Median: 0.283759
  Mean Abs Dev: 0.026136
  IQR: 0.041404

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.066618
     P-value: 0.597156
     Result: ✅ Laplacian
     Fitted location: 0.283872
     Fitted scale: 0.026136
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 16.712 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -2.271 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer2.1.bn1.bias
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: -0.362727
  Variance: 0.006432
  Std: 0.080197
  Min: -0.659952
  Max: -0.058309
  Median: -0.362313
  Mean Abs Dev: 0.061544
  IQR: 0.106723

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.079285
     P-value: 0.377129
     Result: ❌ Not Laplacian
     Fitted location: -0.362123
     Fitted scale: 0.061544
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 1.873 (Laplacian: 3.0)
     P-value: 0.009272
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.198 (Laplacian: 0.0)
     P-value: 0.339805
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.1.conv2.weight
Shape: torch.Size([128, 128, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 147,456

Basic Statistics:
  Mean: -0.006240
  Variance: 0.003130
  Std: 0.055943
  Min: -0.353058
  Max: 0.394301
  Median: -0.002865
  Mean Abs Dev: 0.038684
  IQR: 0.050426

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.060581
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.002864
     Fitted scale: 0.038684
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.639 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.247 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer2.1.bn2.weight
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: 0.241029
  Variance: 0.026950
  Std: 0.164166
  Min: -0.044137
  Max: 0.544365
  Median: 0.243490
  Mean Abs Dev: 0.137480
  IQR: 0.256530

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.101590
     P-value: 0.132852
     Result: ❌ Not Laplacian
     Fitted location: 0.246096
     Fitted scale: 0.137480
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -1.044 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.062 (Laplacian: 0.0)
     P-value: 0.762702
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer2.1.bn2.bias
Shape: torch.Size([128])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 128

Basic Statistics:
  Mean: -0.323180
  Variance: 0.017263
  Std: 0.131387
  Min: -0.652746
  Max: -0.009934
  Median: -0.318411
  Mean Abs Dev: 0.103959
  IQR: 0.182828

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.095046
     P-value: 0.185561
     Result: ❌ Not Laplacian
     Fitted location: -0.315821
     Fitted scale: 0.103959
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 0.038 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.111 (Laplacian: 0.0)
     P-value: 0.589260
     Result: ✅ Symmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

Layer: base.layer3.0.conv1.weight
Shape: torch.Size([256, 128, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 294,912

Basic Statistics:
  Mean: -0.000003
  Variance: 0.001673
  Std: 0.040905
  Min: -0.317468
  Max: 0.417920
  Median: -0.000000
  Mean Abs Dev: 0.021737
  IQR: 0.004710

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.241732
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.021737
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 6.273 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.546 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.bn1.weight
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.150401
  Variance: 0.023929
  Std: 0.154691
  Min: -0.004552
  Max: 0.532291
  Median: 0.130509
  Mean Abs Dev: 0.144986
  IQR: 0.293916

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.251740
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.132332
     Fitted scale: 0.144986
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -1.480 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.298 (Laplacian: 0.0)
     P-value: 0.049627
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.bn1.bias
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: -0.098306
  Variance: 0.012143
  Std: 0.110197
  Min: -0.402057
  Max: 0.004926
  Median: -0.042441
  Mean Abs Dev: 0.094196
  IQR: 0.195827

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.322194
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.040260
     Fitted scale: 0.094196
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.731 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.720 (Laplacian: 0.0)
     P-value: 0.000012
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.conv2.weight
Shape: torch.Size([256, 256, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 589,824

Basic Statistics:
  Mean: -0.000143
  Variance: 0.000154
  Std: 0.012401
  Min: -0.475334
  Max: 0.403146
  Median: -0.000000
  Mean Abs Dev: 0.001505
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.471592
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.001505
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 201.302 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -2.501 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.bn2.weight
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.041355
  Variance: 0.015852
  Std: 0.125906
  Min: -0.000000
  Max: 0.567712
  Median: -0.000000
  Mean Abs Dev: 0.041355
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.500000
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.041355
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 6.754 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 2.883 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.bn2.bias
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.018146
  Variance: 0.003468
  Std: 0.058887
  Min: -0.005036
  Max: 0.301961
  Median: -0.000000
  Mean Abs Dev: 0.018199
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.491714
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.018199
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 9.402 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 3.258 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.downsample.0.weight
Shape: torch.Size([256, 128, 1, 1])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 32,768

Basic Statistics:
  Mean: 0.000126
  Variance: 0.000170
  Std: 0.013031
  Min: -0.238853
  Max: 0.324987
  Median: 0.000000
  Mean Abs Dev: 0.002603
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2280: RuntimeWarning: overflow encountered in exp
  tmp2 = exp(tmp)
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2282: RuntimeWarning: overflow encountered in multiply
  np.sum(tmp*(1.0-tmp2)/(1+tmp2), axis=0) + N]
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2282: RuntimeWarning: invalid value encountered in divide
  np.sum(tmp*(1.0-tmp2)/(1+tmp2), axis=0) + N]
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2286: RuntimeWarning: The iteration is not making good progress, as measured by the 
 improvement from the last ten iterations.
  sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-5)
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.448851
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.000000
     Fitted scale: 0.002603
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 98.829 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 1.700 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.downsample.1.weight
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.004232
  Variance: 0.001579
  Std: 0.039734
  Min: -0.026468
  Max: 0.586511
  Median: -0.000000
  Mean Abs Dev: 0.005097
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.456969
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.005097
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 180.568 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 12.858 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.0.downsample.1.bias
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.018146
  Variance: 0.003468
  Std: 0.058887
  Min: -0.005036
  Max: 0.301961
  Median: -0.000000
  Mean Abs Dev: 0.018199
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.491714
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.018199
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 9.402 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 3.258 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.1.conv1.weight
Shape: torch.Size([256, 256, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 589,824

Basic Statistics:
  Mean: -0.000000
  Variance: 0.000000
  Std: 0.000099
  Min: -0.010948
  Max: 0.013534
  Median: 0.000000
  Mean Abs Dev: 0.000005
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.497785
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.000000
     Fitted scale: 0.000005
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2195.210 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 2.023 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.1.bn1.weight
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: -0.000002
  Variance: 0.000000
  Std: 0.000061
  Min: -0.000565
  Max: 0.000355
  Median: -0.000000
  Mean Abs Dev: 0.000011
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.472656
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000011
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 44.632 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -3.870 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.1.bn1.bias
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.001611
  Variance: 0.000081
  Std: 0.009001
  Min: -0.002765
  Max: 0.074717
  Median: -0.000000
  Mean Abs Dev: 0.001633
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.488280
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.001633
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 37.685 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 6.037 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.1.conv2.weight
Shape: torch.Size([256, 256, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 589,824

Basic Statistics:
  Mean: -0.000000
  Variance: 0.000000
  Std: 0.000193
  Min: -0.027241
  Max: 0.024152
  Median: -0.000000
  Mean Abs Dev: 0.000004
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
/home/latim/PycharmProjects/WatDNN/venv/lib/python3.12/site-packages/scipy/stats/_morestats.py:2286: RuntimeWarning: The iteration is not making good progress, as measured by the 
 improvement from the last five Jacobian evaluations.
  sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-5)
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.497945
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000004
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 6446.656 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -17.464 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.1.bn2.weight
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.000397
  Variance: 0.000020
  Std: 0.004517
  Min: -0.012886
  Max: 0.051755
  Median: -0.000000
  Mean Abs Dev: 0.000639
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.463652
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000639
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 94.261 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 9.126 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer3.1.bn2.bias
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.000579
  Variance: 0.000037
  Std: 0.006065
  Min: -0.058053
  Max: 0.033529
  Median: -0.000000
  Mean Abs Dev: 0.001380
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.487963
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.001380
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 49.013 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -3.732 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.conv1.weight
Shape: torch.Size([512, 256, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 1,179,648

Basic Statistics:
  Mean: 0.000000
  Variance: 0.000001
  Std: 0.000946
  Min: -0.178910
  Max: 0.510746
  Median: -0.000000
  Mean Abs Dev: 0.000006
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.499953
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000006
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 121727.008 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 235.106 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.bn1.weight
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.000494
  Variance: 0.000125
  Std: 0.011170
  Min: -0.000000
  Max: 0.252749
  Median: -0.000000
  Mean Abs Dev: 0.000494
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.500000
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000494
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 507.002 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 22.561 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.bn1.bias
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.000577
  Variance: 0.000170
  Std: 0.013049
  Min: -0.000000
  Max: 0.295267
  Median: -0.000000
  Mean Abs Dev: 0.000577
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.500000
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000577
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 507.002 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 22.561 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.conv2.weight
Shape: torch.Size([512, 512, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 2,359,296

Basic Statistics:
  Mean: -0.000001
  Variance: 0.000000
  Std: 0.000293
  Min: -0.100710
  Max: 0.084183
  Median: -0.000000
  Mean Abs Dev: 0.000001
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.499992
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000001
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 60958.070 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -120.400 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.bn2.weight
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.012970
  Variance: 0.002736
  Std: 0.052308
  Min: -0.117647
  Max: 0.393624
  Median: 0.000000
  Mean Abs Dev: 0.014323
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.486326
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.000000
     Fitted scale: 0.014323
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 17.856 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 4.031 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.bn2.bias
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.015726
  Variance: 0.002598
  Std: 0.050968
  Min: -0.011337
  Max: 0.326176
  Median: -0.000000
  Mean Abs Dev: 0.015958
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.466886
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.015958
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 10.901 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 3.380 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.downsample.0.weight
Shape: torch.Size([512, 256, 1, 1])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 131,072

Basic Statistics:
  Mean: 0.000173
  Variance: 0.000177
  Std: 0.013312
  Min: -0.404363
  Max: 0.569611
  Median: -0.000000
  Mean Abs Dev: 0.001043
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.493945
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.001043
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 410.632 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 7.822 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.downsample.1.weight
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.072928
  Variance: 0.047445
  Std: 0.217818
  Min: -0.001279
  Max: 0.976572
  Median: -0.000000
  Mean Abs Dev: 0.072934
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.496675
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.072934
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 6.895 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 2.882 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.0.downsample.1.bias
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.015726
  Variance: 0.002598
  Std: 0.050968
  Min: -0.011337
  Max: 0.326176
  Median: -0.000000
  Mean Abs Dev: 0.015958
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.466886
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.015958
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 10.901 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 3.380 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.1.conv1.weight
Shape: torch.Size([512, 512, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 2,359,296

Basic Statistics:
  Mean: -0.000001
  Variance: 0.000000
  Std: 0.000440
  Min: -0.189346
  Max: 0.149251
  Median: -0.000000
  Mean Abs Dev: 0.000002
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.499993
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000002
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 78676.734 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -134.820 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.1.bn1.weight
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.000437
  Variance: 0.000098
  Std: 0.009886
  Min: -0.000212
  Max: 0.223693
  Median: -0.000000
  Mean Abs Dev: 0.000438
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.498047
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000438
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 507.000 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 22.561 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.1.bn1.bias
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: -0.000375
  Variance: 0.000072
  Std: 0.008485
  Min: -0.191993
  Max: -0.000000
  Median: -0.000000
  Mean Abs Dev: 0.000375
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.500000
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.000375
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 507.002 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -22.561 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.1.conv2.weight
Shape: torch.Size([512, 512, 3, 3])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 2,359,296

Basic Statistics:
  Mean: -0.000000
  Variance: 0.000000
  Std: 0.000204
  Min: -0.092585
  Max: 0.079237
  Median: 0.000000
  Mean Abs Dev: 0.000001
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.499988
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: 0.000000
     Fitted scale: 0.000001
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 110891.000 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -61.750 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.1.bn2.weight
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: 0.001577
  Variance: 0.000393
  Std: 0.019820
  Min: -0.153594
  Max: 0.168169
  Median: -0.000000
  Mean Abs Dev: 0.004143
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.466429
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.004143
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 33.605 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 3.037 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.layer4.1.bn2.bias
Shape: torch.Size([512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 512

Basic Statistics:
  Mean: -0.000184
  Variance: 0.000042
  Std: 0.006509
  Min: -0.049358
  Max: 0.051884
  Median: -0.000000
  Mean Abs Dev: 0.002060
  IQR: 0.000002

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.429688
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.002060
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 22.781 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -0.098 (Laplacian: 0.0)
     P-value: 0.358163
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.fc.0.weight
Shape: torch.Size([256, 512])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 131,072

Basic Statistics:
  Mean: 0.000972
  Variance: 0.000432
  Std: 0.020794
  Min: -0.411744
  Max: 0.555408
  Median: -0.000000
  Mean Abs Dev: 0.002626
  IQR: 0.000000

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.474366
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.002626
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 138.365 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 7.163 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.fc.0.bias
Shape: torch.Size([256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 256

Basic Statistics:
  Mean: 0.020741
  Variance: 0.001761
  Std: 0.041961
  Min: -0.016723
  Max: 0.198658
  Median: -0.000000
  Mean Abs Dev: 0.021456
  IQR: 0.002410

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.423236
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.021456
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 2.392 (Laplacian: 3.0)
     P-value: 0.047176
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 1.849 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.fc.2.weight
Shape: torch.Size([10, 256])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 2,560

Basic Statistics:
  Mean: -0.013179
  Variance: 0.014272
  Std: 0.119465
  Min: -0.968282
  Max: 0.509725
  Median: -0.000000
  Mean Abs Dev: 0.043056
  IQR: 0.000147

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.310535
     P-value: 0.000000
     Result: ❌ Not Laplacian
     Fitted location: -0.000000
     Fitted scale: 0.043056
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: 16.864 (Laplacian: 3.0)
     P-value: 0.000000
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: -3.050 (Laplacian: 0.0)
     P-value: 0.000000
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 0.0%
     Tests passed: 0/3
============================================================

Layer: base.fc.2.bias
Shape: torch.Size([10])
Type: torch.float32
Requires grad: True
Device: cuda:0
Total parameters: 10

Basic Statistics:
  Mean: -0.011923
  Variance: 0.002060
  Std: 0.045384
  Min: -0.059175
  Max: 0.079951
  Median: -0.012014
  Mean Abs Dev: 0.033395
  IQR: 0.055026

Laplacian Distribution Testing (α=0.5):
  📊 Kolmogorov-Smirnov Test:
     Statistic: 0.244571
     P-value: 0.511751
     Result: ✅ Laplacian
     Fitted location: -0.011812
     Fitted scale: 0.033395
     Note: Parameters estimated from data; p-value may be optimistic.
  📊 Kurtosis Test:
     Excess kurtosis: -0.347 (Laplacian: 3.0)
     P-value: 0.030738
     Result: ❌ Incompatible
  📊 Skewness Test:
     Skewness: 0.784 (Laplacian: 0.0)
     P-value: 0.170032
     Result: ❌ Asymmetric
  🎯 Overall Assessment:
     Likely Laplacian: ❌ NO
     Confidence: 33.3%
     Tests passed: 1/3
============================================================

📋 SUMMARY:
Total layers analyzed: 64
Layers likely following Laplacian distribution: 4
Percentage: 6.2%

🎯 WATERMARKING TARGET ANALYSIS:
🔍 QUICK LAPLACIAN CHECK
========================================
❌ base.conv1.weight: 0.0% confidence
❌ base.layer1.0.conv1.weight: 0.0% confidence
❌ base.layer1.0.conv2.weight: 33.3% confidence
❌ base.layer1.1.conv1.weight: 0.0% confidence
❌ base.layer1.1.conv2.weight: 0.0% confidence
❌ base.layer2.0.conv1.weight: 0.0% confidence
❌ base.layer2.0.conv2.weight: 0.0% confidence
❌ base.layer2.1.conv1.weight: 0.0% confidence
❌ base.layer2.1.conv2.weight: 0.0% confidence
❌ base.layer3.0.conv1.weight: 0.0% confidence
❌ base.layer3.0.conv2.weight: 0.0% confidence
❌ base.layer3.1.conv1.weight: 0.0% confidence
❌ base.layer3.1.conv2.weight: 0.0% confidence
❌ base.layer4.0.conv1.weight: 0.0% confidence
❌ base.layer4.0.conv2.weight: 0.0% confidence
❌ base.layer4.1.conv1.weight: 0.0% confidence
❌ base.layer4.1.conv2.weight: 0.0% confidence
🔍 QUICK LAPLACIAN CHECK
========================================
❌ base.fc.0.weight: 0.0% confidence
❌ base.fc.0.bias: 0.0% confidence
❌ base.fc.2.weight: 0.0% confidence
✅ base.fc.2.bias: 66.7% confidence
"""