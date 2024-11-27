import pandas as pd
import numpy as np

# Function to initialize parameters and constants
def initialize_parameters():
    """
    Define global constants, distributions, and other parameters.
    """
    parameters = {
        # Demographic Parameters
        "age_mean": 35,
        "age_std": 10,
        "age_min": 18,
        "age_max": 75,
        "gender_distribution": {'Male': 0.5, 'Female': 0.5},
        "regions": ['North America', 'Europe', 'Asia', 'South America'],

        # Behavioral Parameters
        "subscription_mean": 12,
        "usage_min": 0,
        "usage_max": 10,
        "complaints_lambda": 2,  # Poisson rate for complaints
        "promotions_used_max": 5,

        # Financial Parameters
        "spending_mean": 50,
        "spending_std": 20,
        "spending_min": 10,
        "spending_max": 200,
        "payment_methods": ['Credit Card', 'PayPal', 'Direct Debit'],
        "device_types": ['Mobile', 'Desktop', 'Tablet'],

        # Churn Parameters
        "base_churn_rate": 0.3,  # Baseline churn probability
    }
    np.random.seed(42)
    return parameters


def generate_demographic_features(num_records, params):
    ages = np.clip(
        np.random.normal(params['age_mean'], params['age_std'], num_records),
        params['age_min'],
        params['age_max']
    ).astype(int)
    genders = np.random.choice(
        list(params['gender_distribution'].keys()),
        size=num_records,
        p=list(params['gender_distribution'].values())
    )
    regions = np.random.choice(params['regions'], size=num_records)

    # Correlate Region and DeviceType
    region_device_prob = {
        'North America': [0.6, 0.3, 0.1],
        'Europe': [0.4, 0.4, 0.2],
        'Asia': [0.3, 0.6, 0.1],
        'South America': [0.5, 0.4, 0.1]
    }
    device_types = [
        np.random.choice(params['device_types'], p=region_device_prob[region])
        for region in regions
    ]

    return pd.DataFrame({"Age": ages, "Gender": genders, "Region": regions, "DeviceType": device_types})


def generate_behavioral_features(num_records, params, demographics):
    """
    Generate behavioral features: SubscriptionLength, UsageFrequency, Complaints, PromotionsUsed, PromotionsDriven.
    Introduce correlations between parameters.
    """
    # Influence Age on SubscriptionLength
    age_effect = (demographics['Age'] < 35).astype(int)  # Younger customers
    subscription_lengths = np.clip(
        np.random.exponential(params['subscription_mean'], num_records) + age_effect * 5,
        1, 24
    ).astype(int)

    # Mark customers likely driven by promotions for short subscriptions
    promotions_driven = (subscription_lengths <= 3) & (np.random.rand(num_records) < 0.8)

    # Influence Age and SubscriptionLength on UsageFrequency
    base_usage = np.random.normal(params['usage_max'] - (demographics['Age'] / 15), 2, num_records)
    subscription_effect = subscription_lengths / 5  # Longer subscriptions increase usage frequency
    usage_frequencies = np.clip(
        base_usage + subscription_effect,
        params['usage_min'], params['usage_max']
    ).astype(int)

    # Complaints: Poisson distribution with correlation to PromotionsUsed
    base_complaints = np.random.poisson(params['complaints_lambda'], num_records)
    complaints = np.clip(base_complaints + (promotions_driven * 1.5), 0, None).astype(int)

    # PromotionsUsed positively correlated with UsageFrequency
    promotions_used = np.clip(
        np.random.randint(0, params['promotions_used_max'] + 1, num_records) +
        (usage_frequencies * 0.1).astype(int),
        0, params['promotions_used_max']
    )

    return pd.DataFrame({
        "SubscriptionLength": subscription_lengths,
        "UsageFrequency": usage_frequencies,
        "Complaints": complaints,
        "PromotionsUsed": promotions_used,
        "PromotionsDriven": promotions_driven.astype(int)
    })

def generate_financial_features(num_records, params, behavioral):
    """
    Generate financial features with correlations to behavioral parameters.
    """
    # Base spending influenced by UsageFrequency and SubscriptionLength
    base_spending = np.random.normal(params['spending_mean'], params['spending_std'], num_records)
    spending = base_spending + (behavioral['UsageFrequency'] * 1.5) - (behavioral['Complaints'] * 2)
    spending = np.clip(spending, params['spending_min'], params['spending_max'])

    # PaymentMethod and DeviceType distribution
    payment_methods = np.random.choice(params['payment_methods'], size=num_records)
    device_types = np.random.choice(params['device_types'], size=num_records)

    return pd.DataFrame({
        "AverageSpending": spending,
        "PaymentMethod": payment_methods,
        "DeviceType": device_types
    })



def generate_churn_labels(features_df, params):
    """
    Generate churn labels based on stronger correlations.
    """
    # Start with a baseline churn probability
    churn_probabilities = np.full(len(features_df), 0.3)

    # Older customers are more likely to churn
    churn_probabilities += (features_df['Age'] > 50) * 0.1

    # High complaints significantly increase churn
    churn_probabilities += (features_df['Complaints'] > 2) * 0.5

    # Short subscription lengths increase churn
    churn_probabilities += (features_df['SubscriptionLength'] < 6) * 0.2

    # High usage frequency and long subscription lengths decrease churn
    churn_probabilities -= (features_df['UsageFrequency'] > 7) * 0.2
    churn_probabilities -= (features_df['SubscriptionLength'] > 12) * 0.15

    # Clip probabilities to a valid range (0 to 1)
    churn_probabilities = np.clip(churn_probabilities, 0, 1)

    # Sample churn labels based on probabilities
    churn_labels = np.random.rand(len(features_df)) < churn_probabilities
    return churn_labels.astype(int)  # Convert True/False to 1/0



def combine_features_into_dataframe(demographics, behavioral, financial):
    return pd.concat([demographics, behavioral, financial], axis=1)



# Main function to generate the dataset
def main():
    num_records = 50000
    params = initialize_parameters()

    # Generate feature sets
    demographics = generate_demographic_features(num_records, params)
    behavioral = generate_behavioral_features(num_records, params, demographics)
    financial = generate_financial_features(num_records, params, behavioral)

    # Combine features
    dataset = combine_features_into_dataframe(demographics, behavioral, financial)

    # Generate churn labels
    dataset['CustomerChurned'] = generate_churn_labels(dataset, params)

    # Save dataset
    dataset.to_csv("..\Different_classification_algorithms\customer_churn.csv", index=False)

    print("Dataset generated and saved as 'customer_churn.csv'.")


if __name__ == "__main__":
    main()
