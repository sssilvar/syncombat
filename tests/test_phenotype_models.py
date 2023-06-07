import os
import pytest


@pytest.mark.parametrize('n_phenotypes', [1, 3, 100])
def test_linear_phenotype_model(n_phenotypes):
    import torch
    from syncombat.phenotypes.models import LinearPhenotypeModel

    # Create a synthetic design matrix
    dm = torch.rand(100, 10)

    # Create a linear phenotype model
    pheno_model = LinearPhenotypeModel(dm, n_phenotypes=n_phenotypes, random_state=42)

    # Assert weights shape
    assert pheno_model.model.weight.shape == (n_phenotypes, dm.shape[1])

    # Assert that the model is identifiable
    pheno_model.assert_model_identifiability()

    # Print the dataframes
    print(pheno_model.dataframe.head())

    # Test save and load using temporary files after loading state_dict for both should be the same
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract state_dict
        state_dict = pheno_model.state_dict()

        # Save model
        pheno_model.save_model(os.path.join(tmpdir, "model.pt"))

        # Instantiate a new model
        pheno_model_new = LinearPhenotypeModel(dm, n_phenotypes=n_phenotypes, random_state=21)

        # Assert weights shape as well
        assert pheno_model.state_dict()['weight'].shape == pheno_model_new.state_dict()['weight'].shape

        # Load model
        pheno_model_new.load_model(os.path.join(tmpdir, "model.pt"))

        # Assert that the keys and elements of state_dicts are the same
        for key in state_dict.keys():
            assert torch.allclose(state_dict[key], pheno_model_new.state_dict()[key])
