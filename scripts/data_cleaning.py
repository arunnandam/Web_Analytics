# Steps I'm following in this script.
# This script is for clean the data, integrating them and making the unified dataset.

# Importing required packages
import numpy as np
import pandas as pd

# Loading the datasets for data cleaning
user_activity_df = pd.read_csv('../data/user_activity_data.csv')
moderator_performance_df = pd.read_csv('../data/moderator_performance_data.csv')
recommendation_data_df = pd.read_csv('../data/recommendation_data.csv')

# Cleaning data

# Function to clean a dataframe
# Steps I'm following - removing duplicates, filling missing values with mean if present for numerical data, mode for categorical data, standardizing datetimes
def clean_dataframe(df, datetime_cols=[]):

    # Droping duplicates
    df = df.drop_duplicates()

    # Handling missing values
    missing_values = df.isnull().sum()
    for col in missing_values.index:
        if missing_values[col] > 0:
            if df[col].dtype in ['float64', 'int64']:
                # handling numerical data
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Handling categorical data
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Standardizing datetime
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

# Before data cleaning
print("User Activity Data", user_activity_df.shape)
print("\nModerator Performance Data", moderator_performance_df.shape)
print("\nRecommendation Data Info", recommendation_data_df.shape)

# Clean each dataset
user_activity_df = clean_dataframe(user_activity_df, datetime_cols=['timestamp'])
moderator_performance_df = clean_dataframe(moderator_performance_df)
recommendation_data_df = clean_dataframe(recommendation_data_df)

# After data cleaning
print("\nUser Activity Data", user_activity_df.shape)
print("\nModerator Performance Data", moderator_performance_df.shape)
print("\nRecommendation Data Info", recommendation_data_df.shape)

# Integrating data

# Left join to retain all user activity sessions
user_recommendation_df = pd.merge(user_activity_df, recommendation_data_df, on='user_id', how='left')

# Using the proportion technique, I will assign moderators to above joined data
# there can be multiple methods like random outer join, average summary metrics join, proportional join
total_user_sessions = len(user_recommendation_df)
total_moderated_sessions = moderator_performance_df['chat_sessions_moderated'].sum()

moderator_performance_df['assigned_sessions'] = (
    (moderator_performance_df['chat_sessions_moderated'] / total_moderated_sessions) * total_user_sessions
).round().astype(int)

# Creating list of moderator IDs based on the assigned sessions
moderator_ids = []
for idx, row in moderator_performance_df.iterrows():
    moderator_ids.extend([row['moderator_id']] * row['assigned_sessions'])

moderator_ids = moderator_ids[:total_user_sessions]
np.random.shuffle(moderator_ids)

# Assigning moderators to above data
user_recommendation_df['moderator_id'] = moderator_ids

# Merge Moderator data
unified_df = pd.merge(user_recommendation_df, moderator_performance_df, on='moderator_id', how='left')

# Exporting the final unifed data
unified_df.to_csv('../data/unified_dataset.csv', index=False)
print("\nUnified dataset has been saved as 'unified_dataset.csv'.")
