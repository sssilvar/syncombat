# A script for generating synthetic age, se, diagnosis, and other covariates for use in synthetic combat.
from functools import lru_cache

import pyro
import pyro.distributions as dist
import torch
import pandas as pd


class CovariateGenerator:
    ALLOWED_LEVELS_OF_NON_IIDNESS = ["low", "medium", "high"]

    age: torch.Tensor
    sex: torch.Tensor
    etiv: torch.Tensor
    site: torch.Tensor

    def __init__(self,
                 n_samples: int,
                 n_sites: int,
                 n_dx_groups: int = 2,
                 sex_non_iidness: str = 'low',
                 age_non_iidness: str = 'low',
                 site_non_iidness: str = 'low',
                 diagnosis_non_iidness: str = 'low',
                 random_state: int = 42
                 ):
        # Assert iidness level is valid
        assert sex_non_iidness in self.ALLOWED_LEVELS_OF_NON_IIDNESS, \
            f"Level of non-iidness must be one of {self.ALLOWED_LEVELS_OF_NON_IIDNESS}."
        assert age_non_iidness in self.ALLOWED_LEVELS_OF_NON_IIDNESS, \
            f"Level of non-iidness must be one of {self.ALLOWED_LEVELS_OF_NON_IIDNESS}."
        assert site_non_iidness in self.ALLOWED_LEVELS_OF_NON_IIDNESS, \
            f"Level of non-iidness must be one of {self.ALLOWED_LEVELS_OF_NON_IIDNESS}."
        assert diagnosis_non_iidness in self.ALLOWED_LEVELS_OF_NON_IIDNESS, \
            f"Level of non-iidness must be one of {self.ALLOWED_LEVELS_OF_NON_IIDNESS}."

        # Initialize parameters
        self.n_samples = n_samples
        self.n_sites = n_sites
        self.n_dx_groups = n_dx_groups

        # Prior parameters for covariate distrubutions: age_mu, age_sigma, sex_p, site_p, dx_p
        age_mu_ab_dict = {'low': (40, 50), 'medium': (40, 60), 'high': (0, 100)}
        age_sigma_gamma_dict = {'low': (10, 0.5), 'medium': (10, 2), 'high': (1, 0.08)}
        site_dirichlet_concentration_dict = {'low': 100, 'medium': 10, 'high': 0.7}
        sex_concentration_dict = {'low': 100, 'medium': 10, 'high': 0.7}
        dx_concentration_dict = {'low': 100, 'medium': 10, 'high': 0.7}

        # Set seed
        self.random_state = random_state
        pyro.set_rng_seed(self.random_state)

        # Initialize non-iidness parameters
        with pyro.plate('site', self.n_sites):
            # Age parameters
            self.age_mu = pyro.sample('age_mu', dist.Uniform(*age_mu_ab_dict[age_non_iidness]))
            self.age_sigma = pyro.sample('age_sigma', dist.Gamma(*age_sigma_gamma_dict[age_non_iidness]))
            print(self.age_mu, self.age_sigma)

            # Sex concentration (male/female) parameters
            sex_concentration = sex_concentration_dict[sex_non_iidness] * torch.ones(2)
            sex_concentration_prior = dist.Dirichlet(sex_concentration)
            self.sex_p = pyro.sample('sex_p', sex_concentration_prior)  # Only one as they add up 1

            # Diagnosis concentration parameters
            dx_concentration = dx_concentration_dict[diagnosis_non_iidness] * torch.ones(self.n_dx_groups)
            dx_proportions_prior = dist.Dirichlet(dx_concentration)
            self.dx_p = pyro.sample('dx_p', dx_proportions_prior)  # Only one as they add up 1

        # Concentration parameter for Dirichlet distribution of site proportions (Dirichlet need to be at least 2D
        site_concentration_param = site_dirichlet_concentration_dict[site_non_iidness] * torch.ones(self.n_sites)
        site_prior = dist.Dirichlet(site_concentration_param)
        self.site_p = pyro.sample('site_p', site_prior)

    @lru_cache(maxsize=1)
    def generate_independent_covariates(self):
        """Generate covariates for n_samples and n_sites."""
        with pyro.plate('subject_j', self.n_samples) as subject_idx:
            site = pyro.sample('site', dist.Categorical(self.site_p))

            # Sample age
            age_mu = self.age_mu[site]  # Mean age at site
            age_sigma = self.age_sigma[site]  # Std of age at site
            age = torch.clip(pyro.sample('age', dist.Normal(age_mu, age_sigma)), 0, 110)

            # Sample sex
            site_sex_p = self.sex_p[site]  # Probability of Male/Female at site
            sex = pyro.sample('sex', dist.Categorical(site_sex_p))

            # Sample diagnosis
            site_dx_p = self.dx_p[site]  # Probability of each diagnosis at site
            dx = pyro.sample('DX', dist.Categorical(site_dx_p))

            # eTIV model
            etiv_mu = torch.tensor([1200, 1100])[sex]  # Male and female eTIV mean
            etiv_loc = etiv_mu + 30 * torch.log(age * 365)
            etiv_sigma = 10 + age * 0.001
            etiv = pyro.sample('eTIV', dist.Normal(etiv_loc, etiv_sigma))

            # Create subject_id
            subject_id = f'SUB_{subject_idx}_SITE_{site}'

        return {'Age': age, 'Sex': sex, 'eTIV': etiv, 'Site': site, 'DX': dx, 'PTID': subject_id}

    @property
    def dataframe(self):
        # Get data as numpy
        data_np = {
            key: val.numpy()
            for key, val in self.generate_independent_covariates().items()
            if torch.is_tensor(val)
        }

        # Define dtypes and create dataframe
        dtype_dict = {'Sex': 'category', 'Age': 'float', 'Site': 'category', 'DX': 'category', 'eTIV': 'float'}
        df = pd.DataFrame(data_np).astype(dtype_dict).reset_index(drop=True)

        # Rename levels to improve readability
        df['Sex'] = df['Sex'].cat.rename_categories({0: 'Male', 1: 'Female'})

        # To rename diagnosis (as it can take multiple values) we estimate the levels and greate the 'Group {i}' names
        dx_levels = df['DX'].cat.categories
        dx_levels_dict = {dx_level: f'DX {i}' for i, dx_level in enumerate(dx_levels)}
        df['DX'] = df['DX'].cat.rename_categories(dx_levels_dict)

        # As done for diagnosis, we rename sites to 'Site {i}'
        site_levels = df['Site'].cat.categories
        site_levels_dict = {site_level: f'Site {i}' for i, site_level in enumerate(site_levels)}
        df['Site'] = df['Site'].cat.rename_categories(site_levels_dict)

        return df
