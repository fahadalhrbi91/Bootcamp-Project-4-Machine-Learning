## Team members:
- Fahad Alhrbi
- Zainab Melaibari
- Majed Alshnifi

# Introduction to the E-commerce Customer Analysis Dataset

In the realm of international e-commerce, understanding customer behavior and satisfaction is crucial for success. Our project focuses on analyzing a customer database from an international e-commerce company that specializes in selling electronic products. By leveraging advanced machine learning techniques, we aim to extract key insights and provide valuable information to the company.

The dataset we're working with contains 10,999 observations of 12 variables, offering a comprehensive view of customer interactions and various operational aspects of the company. Each variable provides specific information about the customers and their experiences, such as warehouse block, mode of shipment, customer care calls, customer rating, cost of the product, prior purchases, product importance, gender, discount offered, weight in grams, and whether the product reached on time.

## Problem Statement

The e-commerce company aims to understand customer behavior when making purchases and assess the frequency of customer care calls. By predicting the number of calls a customer is likely to make, the company can proactively engage with customers and send timely updates about their orders, thereby enhancing customer satisfaction and fostering a positive perception of the company's commitment to customer care.

By analyzing this dataset, we can provide the e-commerce company with valuable insights that will enable them to make data-driven decisions, improve customer satisfaction, and optimize their business processes.

### Objectives

1. Explore the relationship between customer ratings and product delivery timeliness.
2. Analyze the impact of customer care calls on customer satisfaction and repeat purchases.
3. Investigate the influence of product cost and offered discounts on customer behavior.
4. Predict the number of customer care calls based on various customer and transactional factors.
5. Identify the key factors that contribute to higher or lower call volumes.
6. Provide recommendations to optimize customer care strategies and improve overall customer experience.
## Dataset Overview

The dataset contains information about customers and their interactions with an international e-commerce company. Here are the key variables present in the dataset:

- **ID**: ID Number of Customers.
- **Warehouse block**: The warehouse block where products are stored (A, B, C, D, E).
- **Mode of shipment**: The method of shipment used (Ship, Flight, Road).
- **Customer care calls**: The number of calls made for customer care inquiries.
- **Customer rating**: Rating given by customers, ranging from 1 (lowest) to 5 (highest).
- **Cost of the product**: Cost of the product in US dollars.
- **Prior purchases**: Number of prior purchases made by the customer.
- **Product importance**: Categorization of product importance (low, medium, high).
- **Gender**: Gender of the customer (Male or Female).
- **Discount offered**: Discount offered on the specific product.
- **Weight in gms**: Weight of the product in grams.
- **Reached on time**: Target variable indicating whether the product reached on time (0 = reached on time, 1 = not reached on time).

The dataset was obtained from the [Kaggle platform](https://www.kaggle.com/datasets/prachi13/customer-analytics) and forms the basis of our analysis and insights.

## Exploratory Data Analysis (EDA)

To gain insights and understand the dataset, we performed Exploratory Data Analysis (EDA) using various steps and techniques. Here is an overview of the steps we followed:

1. **Head**: We started by examining the first few records in the dataset using the `head` function. This allowed us to get a glimpse of the data and understand its structure.

2. **Shape**: We explored the shape of the dataset using the `shape` attribute, which provided us with the number of rows and columns in the dataset. This information helped us understand the overall size of the dataset.

3. **Info**: We used the `info` function to obtain information about the dataset, including the data types of each column, the number of non-null values, and the memory usage. This helped us understand the data types and identify any potential data inconsistencies.

4. **Describe**: By utilizing the `describe` function, we obtained summary statistics for numerical columns in the dataset. This allowed us to gain insights into the distribution, central tendency, and spread of the numerical data.

5. **Missing Values**: We examined the presence of missing values in the dataset by using the `isnull` function and summing the null values. This step helped us identify any missing data and determine the extent of data completeness.

By performing these steps, we were able to familiarize ourselves with the dataset, understand its structure, and identify any initial patterns or issues that needed further exploration. These insights formed the foundation for our subsequent analysis and decision-making process.
## Additional EDA

In addition to the initial exploratory data analysis (EDA) steps mentioned earlier, we conducted further data exploration and preprocessing. Here are the details of the additional EDA steps:

1. **Dropping Rows**

   We dropped the 'ID' column from the dataset using the `drop` function. This step removed the unnecessary column and reduced the dimensionality of the dataset.

2. **Data Transformation**

   We performed data transformation on certain columns to convert them into numerical values. The specific transformations are as follows:

   - **Gender Column**: We mapped the 'Gender' column to numerical values. We assigned 0 for 'F' (Female) and 1 for 'M' (Male). This transformation allowed us to represent gender as binary values.

   - **Warehouse_block Column**: We mapped the 'Warehouse_block' column to numerical values ranging from 0 to 4. This transformation assigned numeric labels to different warehouse blocks, facilitating further analysis.

   - **Mode_of_Shipment Column**: We mapped the 'Mode_of_Shipment' column to numerical values of 0, 1, and 2. This transformation assigned numeric labels to different modes of shipment (Flight, Ship, and Road) for easier analysis.

   - **Product_importance Column**: We mapped the 'Product_importance' column to numerical values of 0, 1, and 2. This transformation assigned numeric labels to different levels of product importance (low, medium, and high) for better analysis.

3. **Feature Engineering**

   We created a new feature, 'Weight_in_kg', by dividing the 'Weight_in_gms' column by 1000. This transformation allowed us to represent the weight in kilograms, which is a more commonly used unit.
These additional EDA steps enhanced our understanding of the dataset and prepared the data for further analysis and modeling.

# Data Visualization

In this section, we present meaningful insights obtained through data visualization techniques. We have used various charts to analyze the real estate dataset and provide valuable insights.

### Correlation Analysis

To gain insights into the relationships between different variables in the dataset, we performed a correlation analysis. The correlation analysis helps us understand the degree and direction of the linear relationship between pairs of variables. Here is a correlation matrix and a corresponding heatmap visualization:

<center>
    <img src="images/output.png" alt="Dataset Correlation" width="800px">
</center>

The correlation matrix provides a numerical representation of the correlation coefficients between variables. The heatmap visualization offers a visual representation of the correlation matrix, with color intensity indicating the strength of the correlation.


## Regression Machine Learning

In order to predict the number of calls that a customer will make, we applied regression machine learning techniques. Regression models are used to estimate or predict a continuous numerical value based on input features. We compared a total of 15 models and selected the top 3 performing models for further analysis.

The following models were used:

- Support Vector Regressor (SVR) with different kernels
- XGBoost Regressor (XGBRegressor)
- Gradient Boosting Regressor (GradientBoostingRegressor)
- Elastic Net (ElasticNet)
- Lasso (Lasso)
- Ridge (Ridge)
- K-Nearest Neighbors Regressor (KNeighborsRegressor)
- AdaBoost Regressor (AdaBoostRegressor)
- Extra Trees Regressor (ExtraTreesRegressor)
- Gaussian Process Regressor (GaussianProcessRegressor)
- Huber Regressor (HuberRegressor)
- Passive Aggressive Regressor (PassiveAggressiveRegressor)
- Orthogonal Matching Pursuit (OrthogonalMatchingPursuit)
- RANSAC Regressor (RANSACRegressor)
- Random Forest Regressor (RandomForestRegressor)

After evaluating the models, we selected the following three models as the top performers:

- XGBRegressor
- RandomForestRegressor
- ExtraTreesRegressor
### Results
Here are the performance metrics for the selected models:

| Model                  | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Square Error (RMSE) | R^2 Score |
|------------------------|--------------------------|-------------------------|-------------------------------|-----------|
| XGBRegressor           | 0.770938                 | 0.942164                | 0.970651                      | 0.285643  |
| RandomForestRegressor | 0.765282                 | 0.927023                | 0.962820                      | 0.297124  |
| ExtraTreesRegressor    | 0.773268                 | 0.955451                | 0.977472                      | 0.275569  |

<center>
    <img src="images/output3.png" alt="Dataset Correlation" width="800px">
</center>
These models were selected based on their performance in terms of the mean absolute error (MAE), mean squared error (MSE), root mean square error (RMSE), and R^2 score. The lower the values of MAE, MSE, RMSE and there normal distrubtion , the better the model's predictive performance.

These selected models can be used to predict the number of customer calls, providing valuable insights for customer support and engagement strategies.




 [tableau public](https://public.tableau.com/app/profile/fahad.alluqmani/viz/Proejct/Dashboard3?publish=yes)