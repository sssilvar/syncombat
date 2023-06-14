"""
=======================
Biasing submodule following the ComBat model formulation
=======================

This submodule contains the implementation of the ComBat model for batch effect induction. The combat formulation
supposes that the data is generated by a mean effect model with a random intercept and a multiplicative
variance factor for each batch.
"""
from functools import lru_cache

import pandas as pd
import pyro
import torch
import pyro.distributions as dist

from ..covariates.generation import CovariateGenerator
from ..phenotypes.models import PhenotypeModel


class ComBatBiaser:
    """An abstract class for batch effect induction using the ComBat model."""
    y_i: torch.Tensor
    tau_i: torch.Tensor
    gamma_ig: torch.Tensor

    lambda_i: torch.Tensor
    nu_i: torch.Tensor
    delta_ig: torch.Tensor

    sigma_g: torch.Tensor

    def __init__(self, cov_gen: CovariateGenerator, pheno_gen: PhenotypeModel, random_state=42):
        """Abstract class for batch effect induction using the ComBat model.

        Args:
            cov_gen (CovariateGenerator): Covariate generator.
            pheno_gen (PhenotypeModel): Phenotype generator.
            random_state (int): Random seed.
        """
        # Save params
        self.cov_gen = cov_gen
        self.pheno_gen = pheno_gen
        self.random_state = random_state
        self.n_sites = cov_gen.n_sites
        self.n_phenotypes = pheno_gen.n_phenotypes

        # Assert generators are compatible by checking the length of their generated data
        assert len(self.cov_gen.dataframe) == len(self.pheno_gen.dataframe), \
            "The covariate and phenotype generators must be compatible (same length)."

        # Set seed
        torch.manual_seed(self.random_state)

    @lru_cache(maxsize=1)
    def generate_batch_effects(self):
        """Generates batch effects following the ComBat model formulation."""
        # Define generative model for batch effects
        with pyro.plate('genotype', self.n_phenotypes) as g:
            """Plate for each phenotype $g$."""

            with pyro.plate('batch_effects', self.n_sites):
                # Random intercept: gamma_ig ~ N(Y_i, tau_i)
                # Y_i ~ Uniform(0, 0.1)
                # tau_i ~ InversedGamma(0.5, 0.5)
                self.y_i = pyro.sample('y_i', dist.Uniform(0, 0.1))
                self.tau_i = pyro.sample('tau_i', dist.InverseGamma(20, 0.5))
                self.gamma_ig = pyro.sample('gamma_ig', dist.Normal(self.y_i, self.tau_i))

                # Random variance scaling factor: delta_ig ~ Gamma(lambda_i * nu_i, nu_i)
                # lambda_i ~ Gamma(50, 50)
                # nu_i ~ Gamma(50, 1)
                self.lambda_i = pyro.sample('lambda_i', dist.Gamma(50, 50))
                self.nu_i = pyro.sample('nu_i', dist.Gamma(50, 1))
                self.delta_ig = pyro.sample('delta_ig', dist.Gamma(self.lambda_i * self.nu_i, self.nu_i))

            # y_ijg ~ N(mu_ijg, sigma_ijg)
            # mu_ijg = phi_x + gamma_ig
            # sigma_ijg = delta_ig * sigma_g, where sigma_g is the std of the phenotype
            # sigma_g ~ HalfCauchy(0.2)
            # A plate for all the data points
            self.sigma_g = pyro.sample('sigma_g', dist.HalfCauchy(0.2))
            with pyro.plate('data', len(self.cov_gen.dataframe)) as idx:
                # Compute mu_ijg and sigma_g
                phi_x = self.pheno_gen.phi_x[idx[:, None], g]
                site_idx = self.cov_gen.site[idx]

                # Compute mu_ijg and sigma_ijg
                mu_ijg = phi_x + self.gamma_ig[site_idx]
                sigma_ijg = self.delta_ig[site_idx] * self.sigma_g[g]

                return pyro.sample('y_ijg', dist.Normal(mu_ijg, sigma_ijg))

    def full_dataframe(self) -> 'pd.DataFrame':
        """Returns the full dataframe with batch effects."""
        phenotype_tensor = self.generate_batch_effects().numpy()
        phenotype_df = pd.DataFrame(phenotype_tensor, columns=self.pheno_gen.phenotype_names)
        return pd.concat([self.cov_gen.dataframe, phenotype_df], axis=1)

