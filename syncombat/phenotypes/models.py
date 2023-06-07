"""
=================
Phenotypes Models
=================

This submodule contains a variety of models for phenotype generation as a function of covariates.
Among these models there are:

* Linear
* BSplines
* Arbitrary nonlinear models (MLP).

All these models generate the data and store the model parameters in a directory following the PyTorch convention.

Note:
    These simulated phenotypes are considered the ground truth. The functions generated have not yet been biased.

"""
import os

import pandas as pd
import torch


def _assert_path(path: str):
    """Assert that the path exists."""
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")


class PhenotypeModel:
    """An abstract class for phenotype generation models."""
    model: torch.nn.Module

    def __init__(self, tensor_design_matrix, n_phenotypes=1, random_state=42):
        """Abstract class for phenotype generation.

        Args:
            tensor_design_matrix (torch.Tensor): Covariate design matrix of shape (n_samples, n_covariates).
            random_state (int): Random seed.
        """
        # Save prams
        self.in_features = tensor_design_matrix.shape[1]
        self.out_features = n_phenotypes
        self.random_state = random_state
        self.design_matrix = tensor_design_matrix

        # Set seed
        torch.manual_seed(self.random_state)

    def assert_model_identifiability(self):
        """Assert that the model is identifiable."""
        # This is achieved by ensuring that the model's output is zero when the input is zero
        # as the intercepts are captured by the use of the design matrix.

        # Create a zero vector of the same shape as the design matrix
        design_matrix_zero = torch.zeros_like(self.design_matrix)

        # Assert that the model's output is zero when the input is zero
        assert torch.allclose(self.model(design_matrix_zero), torch.zeros(self.out_features))

    def generate_phenotypes(self) -> torch.Tensor:
        pass

    def save_model(self, path: str, overwrite: bool = False):
        """Save model parameters to a directory following the PyTorch convention."""
        # If the path exists and overwrite is False, raise an error
        if os.path.exists(path) and not overwrite:
            raise ValueError(f"Path {path} already exists. Set overwrite=True to overwrite.")

        # Save model parameters
        torch.save(self.model.state_dict(), path)

    def state_dict(self):
        """Return the model parameters."""
        return self.model.state_dict()

    def load_model(self, path: str):
        """Load model parameters from a directory following the PyTorch convention."""
        # Load model parameters
        self.model.load_state_dict(torch.load(path), strict=True)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a dataframe with the generated phenotypes."""
        # Create column names
        col_names = [f"Phenotype_{i}" for i in range(self.out_features)]

        return pd.DataFrame(self.generate_phenotypes().numpy(), columns=col_names)


class LinearPhenotypeModel(PhenotypeModel):
    def __init__(self, tensor_design_matrix, n_phenotypes=1, random_state=42):
        """Linear model class for phenotype generation.

        Args:
            tensor_design_matrix (torch.Tensor): Covariate design matrix of shape (n_samples, n_covariates).
        """

        # Save prams
        super().__init__(tensor_design_matrix, n_phenotypes, random_state)

        #  Instantiate model
        # Bias is set to zero given that the intercept is already included in the design matrix
        self.model = torch.nn.Linear(self.in_features, self.out_features, bias=False)

        # Generate phenotype
        self.generate_phenotypes()

    def generate_phenotypes(self):
        with torch.no_grad():
            return self.model(self.design_matrix)
