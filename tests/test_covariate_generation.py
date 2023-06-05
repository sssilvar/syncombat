import torch

from syncombat.covariates.generation import CovariateGenerator


def test_covariate_generation():
    # Initialize CovariateGenerator with test parameters
    n_samples = 1000
    n_sites = 5
    n_dx_groups = 2
    non_iid = "medium"
    generator = CovariateGenerator(n_samples, n_sites, n_dx_groups)
    df = generator.dataframe

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a grid with 1 row and 3 columns for subplots
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3)

    # Subplot 1: Age Distribution per Site
    ax1 = plt.subplot(gs[0, 0])
    sns.violinplot(x='Site', y='Age', data=df, ax=ax1)
    ax1.set_title('Age Distribution per Site')

    # Subplot 2: Sex Distribution per Site
    ax2 = plt.subplot(gs[0, 1])
    sns.countplot(x='Site', hue='Sex', data=df, ax=ax2)
    ax2.set_title('Sex Distribution per Site')

    # Subplot 3: Diagnosis Distribution per Site
    ax3 = plt.subplot(gs[0, 2])
    sns.countplot(x='Site', hue='DX', data=df, ax=ax3)
    ax3.set_title('Diagnosis Distribution per Site')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    sns.lmplot(df, x='Age', y='eTIV', hue='Sex', order=2)
    plt.suptitle('eTIV vs Age (all sites)')
    plt.show()

