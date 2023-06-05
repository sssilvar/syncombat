import torch


def etiv_model(age, sex):
    """An empirical model of eTIV that accounts for sex and age dependency.

    Args:
        age (torch.Tensor): Age in years.
        sex (torch.Tensor): Sex indicator (1: Male, 0: Female)
    """
    # Recode sex to -1 and 1
    sex = torch.where(sex == 1, 1, -1).float().reshape(sex.shape)

    # Define the coefficients
    intercept = 1200
    age_coefficient = 30
    sex_coefficient = 20

    # Transform age using logarithmic function
    log_age = torch.log(age * 365)

    # Calculate the linear combination of variables
    linear_combination = intercept + age_coefficient * log_age + sex_coefficient * sex

    # Define age and sex-dependent noise (variability) functions
    age_noise_std = lambda age: 1 + age * 0.001
    sex_noise_std = lambda sex: 30 + torch.sigmoid(sex) * 70

    # Generate age-dependent and sex-dependent noise (variability)
    age_noise = torch.normal(mean=0, std=age_noise_std(age))
    sex_noise = torch.normal(mean=0, std=sex_noise_std(sex))

    # Calculate the eTIV value with age and sex-dependent noise (variability)
    etiv = linear_combination + age_noise * log_age + sex_noise * sex
    return etiv
