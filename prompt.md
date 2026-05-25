PROMPT
You are a Data scientist , and you are developing an application for e-commerce and retail platforms to build Volatility aware  AI sales forecasting system for an e-commerce and retail management system to reduce inventory waste and improve supply chain efficiency.
Context and Role                                                                                                                                                                                                                                                            
As a Data Scientist you will be developing the retail analytics platform . You need to design an interpretable and scalable volatility-based system for forecasting sales in order to minimize the difference between real forecasting and predicted forecasting. The solution will be expected to handle classical e-commerce transaction data, perform feature engineering, build a hybrid machine learning model pipeline , and produce a daily forecast of sales.
Objective 
Develop an end-to-end Python program to consume structured sales data from e-commerce transactions, conduct feature engineering on this data, build an ensemble XGBoost & ARIMA prediction pipeline, utilize SHAP to perform feature selection and generate reports with an interactive UI along with visualizations.
Input data
Transaction dataset
date, product_id, category, sales_quantity, price, promotional_flag, day_of_week, month
Data Preprocessing Specifications
Data Cleaning
•	Process the dataset to clean any missing values, duplicates, and invalid dates.
•	Sort the dataset on a chronological basis using product_id.
•	Create the following features:

	Lag features: t-1, t-7, t-14
	Rolling averages: 7-day and 14-day rolling averages and standard deviation
	Volatility: rolling standard deviation of variable-length windows (default: 7 days)
	Seasonal factors: day of week, month, whether or not it’s a weekend
	Promotion interactions: promotion flag * lag features


•	Use forward fill for small missing values but ignore data points with many consecutive missing values.
•	Features must be aligned on the same time basis.

Model Requirenments
Build a pipeline of a sequential hybrid model:
1. XGBoost Predictive Model Component

Train an XGBoost regressor with all the engineered features.
Split data for training, validation, and test sets using time-based partitioning only (without randomizing).do not assume prepare data according to the intstructions mentioned 
Tuning hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree.

2. SHAP-Driven Feature Selection

Estimate SHAP values for the XGBoost model built above.
Eliminate features according to a defined threshold for SHAP values; then re-train.
Log selected/deselected features.

3. ARIMA Correction of Residuals

Estimating residuals: r_t = y_t - ŷ_ML
Train an ARIMA on residuals to predict them; r̂_t will be predictions for these residuals.
The final prediction: ŷ_final = ŷ_ML + r̂_t

4. Baselines

Develop ARIMA, SARIMA, and Na
Requirements for Evaluation

Models will be evaluated using time-based validation on the held-out test set.
Measures to be provided: RMSLE (primary), RMSE, and R² for each model.
Generate a comparative table and determine the best model using RMSLE.
UI requirenments
To build an interactive web dashboard using streamlit with the following components and structure
1.	File upload : where user can upload the csv dataset, display results in the form of chart and tables.
Display a sales trend line using charts for the product_id that user selects.
       2.Model Training: Include sidebars to forecast horizon ,volatility window size ,SHAP pruning        threshold, and ARIMA order .
A run button that clicks the whole pipeline.
3.Results & Forecasting: Chart comparinig, actual sales vs baseline model prediction ,and hybrid model forecasts.Add model comparision and tables .
4. Expalanaibility: SHAP feature explanation 
Rolling volatility
API Section 
You need to build a REST API using FastAPI with the following endpoints:
-POST /upload-aacepts a csv file 
-POST/ train-it should trigger the full forecasting pipeline 
-GET /forecast/{product_id} -returns forecast JSON for a specific product id 
-GET /metrices -return model evaluation last metrices
-endpoints should must return structuted JSON responses.
-include request validation 
-API must run independently from the ui streamlit 

Output Deliverables
Visualizations that is saved in png 
Charts (stored in PNG files):
Comparison of actual and predicted sales across all models
Rolling volatility pattern for each item
Top 15 SHAP feature importance plot
Series plots: residuals (actual versus fitted using ARIMA model)

Code structure and Documentations 

Organize everything into separate modules 
Data loader file: ingestion and validation
Python.py file: basic python file with baseline models 
Feature Engineering file: it should contain all the feature explainability
Models.py: containd model definitions
Csv files:should contain all datsets 
Evaluation .py : it should contain metrices and comparison tables 
Visualization.py: all plots should be included 
Reporter.py: JSON files 
app.py: UI dashboard
main.py: CLI ,pipeline execution

Error Handling
Handles: Missing files, Invalid CSV records, Convergence problems in ARIMA modeling, Errors in SHAP calculations, Incorrect date formats. Uses configurable logging (DEBUG, INFO, WARNING, ERROR)and display user friendly error message not trackables.

Performance and Scalability
Can handle datasets containing up to 5 million daily observations from thousands of SKUs.
Uses vectorization via pandas/numpy, not iterative processing row by row.
Random seed initialization across all modules ensures repeatability of results.
Caches the processed data (Parquet format) to avoid unnecessary reprocessing.
Models should be organized such that other regressors (LightGBM) or correctors (LSTM) could be easily added.
Tools/Libraries
Pandas, Numpy, Sklearn, XGBoost, Statsmodels, SHAP, Matplotlib, Seaborn, Argparse, Logging, Json,streamlit argparse,logging,json
