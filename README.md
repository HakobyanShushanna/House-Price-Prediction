House Price Prediction Analysis

Overview
This repository contains a Jupyter notebook (house_price_prediction.ipynb) that performs analysis and predictions on house prices using various machine learning regression models. The project explores a dataset sourced from Kaggle, focusing on features such as area, number of bedrooms, bathrooms, and amenities like air conditioning and parking availability to predict house prices accurately.

Dataset
The dataset (Housing.csv) consists of 545 entries with 13 columns:

price: Price of the house
area: Area of the house in square feet
bedrooms: Number of bedrooms
bathrooms: Number of bathrooms
stories: Number of stories
mainroad: Binary feature indicating presence of a main road
guestroom: Binary feature indicating presence of a guest room
basement: Binary feature indicating presence of a basement
hotwaterheating: Binary feature indicating presence of hot water heating
airconditioning: Binary feature indicating presence of air conditioning
parking: Number of parking spots
prefarea: Binary feature indicating preferred location
furnishingstatus: Furnishing status categorized as furnished, semi-furnished, or unfurnished

Installation
To run the notebook locally, ensure you have Python 3.x installed along with Jupyter Notebook. Install the necessary Python packages using:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

Usage
Clone the repository:
git clone https://github.com/HakobyanShushanna/House-Price-Prediction.git
cd House-Price-Prediction

Open and run the Jupyter notebook:
jupyter notebook house_price_prediction.ipynb

Follow the instructions in the notebook to explore the data, preprocess it, train machine learning models, and predict house prices.

Data Preprocessing
The dataset underwent preprocessing steps including:

Conversion of categorical variables to numerical representations
Scaling of numerical features using StandardScaler
Splitting into training and testing sets
Data Visualization
The notebook includes visualizations such as:

Distribution of house prices
Correlation heatmap of features
Model Training and Evaluation
Four regression models were evaluated:

Linear Regression
Decision Tree
Random Forest
XGBoost
Evaluation metrics used:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²) Score
Hyperparameter Tuning
Additionally, Ridge, Lasso, and ElasticNet regression models underwent hyperparameter tuning using GridSearchCV to optimize performance.

Results
Linear Regression demonstrated the best performance with an RMSE of approximately 1,331,071 and an R² score of 0.649, indicating a good fit to the data. Features such as area, number of bedrooms, and air conditioning were found to be significant predictors of house prices.

Conclusion
In conclusion, this project successfully explored and analyzed factors influencing house prices using machine learning techniques. The findings underscore the importance of specific features in predicting house prices accurately.
