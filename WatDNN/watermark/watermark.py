import abc
import torch
import os

from util.metric import Metric


# Corrected and coherent Watermark abstract class
class Watermark(abc.ABC):
    """
    Abstract base class for watermarking defenses.
    """

    def __init__(self):
        """
        Create a watermarking defense object
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """
        Return the name of the watermarking method

        Returns:
            str: Name of the watermarking method
        """
        raise NotImplementedError

    @abc.abstractmethod
    def keygen(self, **kwargs):
        """
        Generate a secret key and message.

        Returns:
            dict: Dictionary containing watermarking keys and metadata
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, init_model, test_loader, train_loader, config):
        """
        Embed a message into a model using a secret key.

        Args:
            init_model: Initial model to be watermarked
            test_loader: DataLoader for test data
            train_loader: DataLoader for training data
            config: Configuration dictionary

        Returns:
            tuple: (watermarked_model, embedding_metrics)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, classifier, supp):
        """
        Extract a message from the model given the watermarking key.

        Args:
            classifier: Watermarked classifier model
            supp: Dictionary containing all information about the watermarking key

        Returns:
            tuple: (extracted_message, extraction_metrics)
        """
        raise NotImplementedError


    @staticmethod
    def save(path: str = None, supplementary: dict = None):
        """ Persist this instance without watermarking keys.
        This is a default loader that should be overridden by the subclass.
        param path The path to the filename ('.pth') to save this defense. to which this defense is saved.
        param supplementary Data to save as dictionary.
        """

        if not path.endswith('.pth'):
            print(f"[WARNING] Defense instance saved as '{path}', but should end in '.pth'.")

        print(f"Saving defense instance at {path}")
        savedir = os.path.dirname(path)
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)

        torch.save(supplementary, path)


    @staticmethod
    def load(path=None):
        """ Load this instance.
        param path The path to The filename ('.pth') to load this defense to which this defense is saved.
        """
        if not path.endswith('.pth'):
            print(f"[WARNING] Defense instance loaded from a '{path}', which should end in '.pth'.")

        print(f'Loading model...at {path}')
        return torch.load(path)


    @staticmethod
    def _get_ber(ext_watermark, watermark):
        # print(ext_watermark)
        wat_ext = (ext_watermark > 0.5) * 1.
        ber = float(Metric.get_ber(wat_ext.cpu(), watermark.cpu()))

        return ber
