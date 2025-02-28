
import numpy as np
import pandas as pd

unified_df = pd.read_csv('../data/unified_dataset.csv')

# Metrics1 - Per User Engagement Score
# This metric is used to calculate engagement score by giving importance to session length and message length and giving importance to feedback rating
# Use the 95th percentile to handle outliers globally
session_length_95 = unified_df['session_length'].quantile(0.95)
messages_sent_95 = unified_df['messages_sent'].quantile(0.95)

# Cap session length and messages at the 95th percentile
unified_df['capped_session_length'] = np.where(
    unified_df['session_length'] > session_length_95, session_length_95, unified_df['session_length']
)

unified_df['capped_messages_sent'] = np.where(
    unified_df['messages_sent'] > messages_sent_95, messages_sent_95, unified_df['messages_sent']
)

# Calculate Engagement Score per session
unified_df['engagement_score'] = (
    (unified_df['capped_session_length'] / session_length_95) * 0.4 +
    (unified_df['capped_messages_sent'] / messages_sent_95) * 0.4 +
    (unified_df['feedback_rating'] / 5) * 0.2
)

# Aggregate Engagement Score per user (average across sessions)
user_engagement = unified_df.groupby('user_id')['engagement_score'].mean().reset_index()
user_engagement.rename(columns={'engagement_score': 'avg_engagement_score'}, inplace=True)

# Per User Responsiveness Efficiency
# This metric is used to calclulate responsiveness efficieny by dividing avg_reponse_time and user_satisfaction_score. Finally diving by 5 to normalize it.
unified_df['responsiveness_efficiency'] = (
    (1 / (unified_df['avg_response_time'] + 1)) * (unified_df['user_satisfaction_score'] / 5)
)
user_responsiveness = unified_df.groupby('user_id')['responsiveness_efficiency'].mean().reset_index()
user_responsiveness.rename(columns={'responsiveness_efficiency': 'avg_responsiveness_efficiency'}, inplace=True)

# Per User Refined RCR
# This metric is used to calculate the recommendations engaged per total_recommendations
users_with_recommendations = unified_df[~unified_df['recommendation_id'].isnull()]
median_feedback_score = users_with_recommendations['feedback_score'].median()

# Identify engaged recommendations: click_through_rate > 0 and feedback_score > median
users_with_recommendations['engaged_recommendation'] = (
    (users_with_recommendations['click_through_rate'] > 0) & 
    (users_with_recommendations['feedback_score'] > median_feedback_score)
).astype(int)

# Calculate RCR per user
user_rcr = users_with_recommendations.groupby('user_id')['engaged_recommendation'].sum().reset_index()
total_recommendations_per_user = users_with_recommendations.groupby('user_id')['recommendation_id'].nunique().reset_index()
total_recommendations_per_user.rename(columns={'recommendation_id': 'total_recommendations'}, inplace=True)

# Merge engaged counts with total recommendations
user_rcr = pd.merge(user_rcr, total_recommendations_per_user, on='user_id', how='left')
user_rcr['refined_rcr'] = (user_rcr['engaged_recommendation'] / user_rcr['total_recommendations']) * 100
user_rcr = user_rcr[['user_id', 'refined_rcr']]

# Creating Metrics Dataset

# Merge all user-level metrics into a single DataFrame
user_metrics = user_engagement.merge(user_responsiveness, on='user_id', how='left')
user_metrics = user_metrics.merge(user_rcr, on='user_id', how='left')

# Fill any missing RCR values with 0 (for users without recommendations)
user_metrics['refined_rcr'] = user_metrics['refined_rcr'].fillna(0)

# Optional: Export the per-user metrics to CSV
user_metrics.to_csv('user_level_metrics.csv', index=False)
print("\nUser-level metrics have been saved as 'user_level_metrics.csv'.")
