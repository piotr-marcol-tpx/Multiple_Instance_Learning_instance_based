from torch.utils import data
import torch
import numpy as np
from transform_pipelines import pipeline_transform_soft, pipeline_transform_paper, pipeline_transform, preprocess
from train_utils import H_E_Staining
import h5py


class HDF5Dataset(data.Dataset):
    def __init__(
            self,
            file_path,
            mode,
            images_dataset_name="patches",
            domain_gt_dataset_name="domain_oh",
            tissue_gt_dataset_name="tissue_oh",
    ):
        self.file_path = file_path
        self.mode = mode
        self.images_dataset_name = images_dataset_name
        self.domain_gt_dataset_name = domain_gt_dataset_name
        self.tissue_gt_dataset_name = tissue_gt_dataset_name
        self.length = None
        self._open_hdf5()

        with h5py.File(self.file_path, 'r') as hf:
            self.length = hf[images_dataset_name].shape[0]

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        patch = self._hf[self.images_dataset_name][index]
        domain_oh = self._hf[self.domain_gt_dataset_name][index]
        # tissue_oh = self._hf[self.tissue_gt_dataset_name][index]

        h_e_matrix = np.array([0, 0, 0, 0, 0, 0])

        if self.mode == 'train':
            k = patch
            """ not using staning processing
            b = False
            while not b:
                k = pipeline_transform_soft(image=k)['image']
                try:
                    h_e_matrix = H_E_Staining(k)
                    b = True
                except:
                    pass
                    # k = pipeline_transform_soft(image=k)['image']
            
            h_e_matrix = np.reshape(h_e_matrix, 6)
            h_e_matrix = np.asarray(h_e_matrix)
            """
            q = pipeline_transform_paper(image=k)['image']

        else:
            k = patch
            q = pipeline_transform(image=k)['image']

        # data transformation
        q = preprocess(q).type(torch.FloatTensor)
        k = preprocess(k).type(torch.FloatTensor)
        # h_e_matrix = torch.FloatTensor(h_e_matrix)

        # return k, q, h_e_matrix, domain_oh
        return k, q, domain_oh


class BalancedMultimodalSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None, alpha=0.5):
        print("Creating sampler...")
        self.classes_oh = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        class_sample_count = np.sum(self.classes_oh, axis=0)
        weights = np.sum(self.classes_oh / class_sample_count, axis=1)

        self.weights_original = torch.DoubleTensor(weights)
        self.weights_uniform = np.repeat(1 / self.num_samples, self.num_samples)

        beta = 1 - alpha
        self.weights = (alpha * self.weights_original) + (beta * self.weights_uniform)

    def _get_label(self, idx):
        return np.argmax(self.classes_oh[idx])

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalancedMutlimodalSampler(BalancedMultimodalSampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedMutlimodalSampler, self).__init__(dataset, indices, num_samples)
        self.weights = self.weights_original


class BalancedBinarySampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None, alpha=0.5):
        print("Creating sampler...")
        self.classes_oh = dataset
        # binary class - whether image comes from pubmed articles
        self.binary_class = self.classes_oh[:, -1]
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        class_sample_count = {1: self.binary_class.sum()}
        class_sample_count.update({0: self.num_samples - class_sample_count[1]})
        weights = np.array([1/class_sample_count[class_] for class_ in self.binary_class])

        self.weights_original = torch.DoubleTensor(weights)
        self.weights_uniform = np.repeat(1 / self.num_samples, self.num_samples)

        beta = 1 - alpha
        self.weights = (alpha * self.weights_original) + (beta * self.weights_uniform)

    def _get_label(self, idx):
        return self.binary_class[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalancedBinarySampler(BalancedBinarySampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedBinarySampler, self).__init__(dataset, indices, num_samples)
        self.weights = self.weights_original


class DatasetBag(data.Dataset):

    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        WSI = self.list_IDs[index]

        return WSI
