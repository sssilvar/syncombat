def test_combat_biaser():
    import numpy as np
    from syncombat.biasing.combat import ComBatBiaser
    from syncombat.covariates.generation import CovariateGenerator
    from syncombat.phenotypes.models import LinearPhenotypeModel

    # Create a covariate generator
    covariate_generator = CovariateGenerator(n_samples=100, n_sites=3, random_state=42)

    # Create a phenotype model
    phenotype_model = LinearPhenotypeModel(
        covariate_generator.tensor_design_matrix,
        n_phenotypes=3,
        random_state=42
    )

    # Create a ComBat biaser
    combat_biaser = ComBatBiaser(cov_gen=covariate_generator, pheno_gen=phenotype_model, random_state=42)
    df = combat_biaser.full_dataframe()

    # Correct covariates and perform a statistical test to
    # assert that the batch effects are present (properly biased).
    # Using ANCOVA for now.
    # Example on how the dataframe looks:
    #
    # |    |     Age | Sex   |    eTIV | Site   | DX   |   Phenotype_0 |   Phenotype_1 |   Phenotype_2 |
    # |---:|--------:|:------|--------:|:-------|:-----|--------------:|--------------:|--------------:|
    # |  0 | 46.4709 | Male  | 1489.98 | Site 2 | DX 1 |      -3.93727 |     -0.375784 |     -0.176476 |

    # Perform ANCOVA
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    p_values = []

    # For each phenotype, perform ANCOVA
    for phenotype_name in phenotype_model.phenotype_names:
        # Correct for covariates: Age, Sex, eTIV and DX leaving site for testing its presence
        model = smf.ols(f'{phenotype_name} ~ Age + Sex + eTIV + DX', data=df).fit()

        # Extract the residuals from the model and add them to the dataframe
        df[f'{phenotype_name}_resid'] = model.resid

        # Drop the covariate columns and keep only the site and phenotype residual columns
        data = df[['Site', f'{phenotype_name}_resid']]

        # Perform ANCOVA
        model = smf.ols(f'{phenotype_name}_resid ~ Site', data=data).fit()
        anova_res = sm.stats.anova_lm(model, typ=2)

        # Extract p_value
        p_value = anova_res['PR(>F)'][0]

        # Append p-value to list
        p_values.append(p_value)
        print(f'p-value for {phenotype_name}: {anova_res["PR(>F)"][0]}')

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Plot the residuals' phenotype distribution hue by site
        g = sns.displot(data=df, x=f'{phenotype_name}_resid', hue='Site', kind='kde')

        # Add the p-value to the plot
        plt.text(x=0.5, y=0.5, s=f'p-value: {p_value:.5f}', transform=g.ax.transAxes)
        plt.title(f'{phenotype_name} residuals distribution')
        plt.show()

    # Assert at least one p-value is below 0.05
    assert np.any(np.array(p_values) < 0.05)
