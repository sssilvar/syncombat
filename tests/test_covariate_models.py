def test_etiv_model():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import torch

    from syncombat.covariates.models import etiv_model

    # Generate age and sex values for testing
    # ages = torch.linspace(torch.tensor(90).log(), torch.tensor(365*95).log(), 2**4)
    ages = torch.linspace(1, 90, 200)
    sex = torch.distributions.Binomial(probs=torch.tensor(0.5)).sample(torch.Size((len(ages), )))
    # Assuming 0 for female and 1 for male

    samples = []
    # Repeat the sampling and dataframe creation process multiple times
    num_samples = 30  # Number of samples to generate
    for _ in range(num_samples):
        # Generate eTIV values using the model
        etiv_values = torch.tensor([etiv_model(age, sex) for age, sex in zip(ages, sex)])

        # Create a temporary dataframe for each sample
        temp_df = pd.DataFrame({"Age": ages.numpy(), "eTIV": etiv_values.numpy(), 'Sex': sex.numpy()})

        # Append the temporary dataframe to the list of samples
        samples.append(temp_df)
    df = pd.concat(samples, axis=0)

    # Plot eTIV vs Age using seaborn
    sns.lineplot(x="Age", y="eTIV", hue='Sex', data=df)
    plt.show()

