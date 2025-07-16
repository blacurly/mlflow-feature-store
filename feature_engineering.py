import pandas as pd

def create_feature_v1(df):
    df = df.copy()
    df['total_lift'] = df[['candj', 'snatch', 'deadlift']].sum(axis=1)
    df = df[df['weight'] != 0]
    df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()
    return df[['age', 'height', 'weight', 'total_lift']]

def create_feature_v2(df):
    df = df.copy()
    df['total_lift'] = df[['candj', 'snatch', 'deadlift']].sum(axis=1)
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Avoid division by zero
    df = df[df['weight'] != 0]
    df['lift_per_kg'] = df['total_lift'] / df['weight']

    # Remove any rows with NaN or infinite values
    df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()

    return df[['age', 'BMI', 'lift_per_kg', 'total_lift']]

