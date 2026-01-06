# Identifying Customer Patterns Influencing Hotel Bookings

## Overview
This project analyzes customer behavior on Expedia to identify distinct traveler segments and understand the factors that influence hotel booking decisions. Using unsupervised and supervised machine learning techniques, we segment customers and model booking likelihood to support targeted marketing and pricing strategies.

## Problem Statement
Online travel platforms face the challenge of converting search behavior into completed bookings. Customers differ widely in price sensitivity, travel purpose, group size, and responsiveness to promotions. This project aims to:
- Identify distinct customer segments based on search and booking behavior
- Understand how booking drivers differ across segments
- Provide actionable insights to improve targeting, promotions, and conversion rates

## Data
- Source: Kaggle – Expedia Hotel Booking Behavior dataset
- Original size: ~9.9 million search records
- Sampled: 1 million rows using stratified sampling to maintain booking distribution
- Target Variable: `booking_bool` (booked vs. not booked)
- Key Features:
  - Price, length of stay, distance to destination
  - Number of adults, children, and rooms
  - Hotel star rating and review score
  - Promotion exposure
  - Saturday-night stay indicator

## Data Processing & Feature Preparation
- Removed null values and irrelevant variables
- Applied extreme outlier capping to preserve dataset integrity
- Corrected skewness using PowerTransformer (Yeo-Johnson)
- Standardized numeric features for distance-based modeling
- Retained binary indicators for behavioral interpretation

## Modeling Approach

### Customer Segmentation (Unsupervised Learning)
- Applied **K-Means clustering**
- Determined optimal cluster count (k = 4) using the Elbow Method and KneeLocator
- Visualized cluster separation using PCA
- Analyzed cluster profiles to define traveler segments

### Booking Prediction (Supervised Learning)
- Built **Logistic Regression models** for each customer segment
- Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices
- Compared segment-level performance to identify high-confidence booking patterns

## Key Results & Insights
- Four distinct customer segments emerged:
  - Solo nearby travelers
  - Price-sensitive budget travelers
  - Adult-only, quality-seeking travelers
  - Large group and family travelers
- Room count and children count were the strongest drivers of segmentation
- Promotion sensitivity varied significantly by segment
- Certain segments showed substantially higher recall, enabling more reliable booking prediction

## Business Implications
- Enables targeted promotions based on traveler type
- Supports dynamic pricing and personalized marketing
- Helps prioritize high-conversion customer segments
- Improves efficiency of marketing spend by segment-specific strategies

## Limitations
- Large dataset required sampling, which may limit edge-case behavior
- Overlapping feature patterns reduced separation for some clusters
- Logistic regression limits modeling of complex non-linear interactions

## Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · kneed

## My Contributions
- Developed the full data pipeline and analytical workflow, including data cleaning, transformation, clustering, and modeling
- Implemented K-Means clustering, PCA visualization, and all Logistic Regression models in Python
- Led and authored the analysis and results sections, including customer segmentation insights and model interpretation

## Contributors
- **Tyler Camelot**
- Ziqi Huang
- Yu-Fang Huang
- Ilnaz Bagheri
- Lasya Reddy Kotha
