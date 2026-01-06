import pandas as pd
df = pd.read_csv("nyc_housing_base.csv")
df.head()

#let's take a look at the dataset
  print("DataFrame Shape:", df.shape)
print("\nDataFrame Info:")
df.info()
print("\nDescriptive Statistics:")
df.describe()

#check the missing values
  print("Missing values before handling:")
print(df.isnull().sum())

#fill the missing values with a median
# List of numerical columns to impute with median
numerical_cols_to_impute = ['zip_code', 'resarea', 'comarea', 'numfloors', 'latitude', 'longitude', 'landuse']

# Impute missing values with the median for the specified numerical columns
for col in numerical_cols_to_impute:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after handling:")
print(df.isnull().sum())

#sales price distribution graph
import matplotlib.pyplot as plt

# Create a histogram of the 'sale_price' column
plt.figure(figsize=(10, 6))
plt.hist(df['sale_price'], bins=50, color='skyblue', edgecolor='black', label='Sale Price Distribution')
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()

# Calculate and print descriptive statistics for 'sale_price'
print("\nDescriptive Statistics for Sale Price:")
print(f"Mean: {df['sale_price'].mean():,.2f}")
print(f"Median: {df['sale_price'].median():,.2f}")

#correlation matrix with sales price
import seaborn as sns

# Select only numerical columns
numerical_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Extract correlations with 'sale_price'
sale_price_correlations = correlation_matrix[['sale_price']].sort_values(by='sale_price', ascending=False)

# Create a heatmap of the correlations with 'sale_price'
plt.figure(figsize=(8, 10))
sns.heatmap(sale_price_correlations, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix with Sale Price')
plt.show()
print(f"Standard Deviation: {df['sale_price'].std():,.2f}")

#trying to find relations with price

# Set up the figure and axes for subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Scatter plot for 'building_age' vs 'sale_price'
sns.scatterplot(ax=axes[0, 0], x=df['building_age'], y=df['sale_price'], label='Building Age', alpha=0.6)
axes[0, 0].set_title('Sale Price vs. Building Age')
axes[0, 0].set_xlabel('Building Age')
axes[0, 0].set_ylabel('Sale Price')
axes[0, 0].legend()
axes[0, 0].ticklabel_format(style='plain', axis='y')

# Scatter plot for 'bldgarea' vs 'sale_price'
sns.scatterplot(ax=axes[0, 1], x=df['bldgarea'], y=df['sale_price'], label='Building Area', alpha=0.6)
axes[0, 1].set_title('Sale Price vs. Building Area')
axes[0, 1].set_xlabel('Building Area')
axes[0, 1].set_ylabel('Sale Price')
axes[0, 1].legend()
axes[0, 1].ticklabel_format(style='plain', axis='y')

# Scatter plot for 'unitsres' vs 'sale_price'
sns.scatterplot(ax=axes[1, 0], x=df['unitsres'], y=df['sale_price'], label='Residential Units', alpha=0.6)
axes[1, 0].set_title('Sale Price vs. Residential Units')
axes[1, 0].set_xlabel('Residential Units')
axes[1, 0].set_ylabel('Sale Price')
axes[1, 0].legend()
axes[1, 0].ticklabel_format(style='plain', axis='y')

# Box plot for 'borough_x' vs 'sale_price'
sns.boxplot(ax=axes[1, 1], x=df['borough_x'], y=df['sale_price'], hue=df['borough_x'], palette='viridis', legend=False)
axes[1, 1].set_title('Sale Price Distribution by Borough')
axes[1, 1].set_xlabel('Borough (Numeric ID)')
axes[1, 1].set_ylabel('Sale Price')
axes[1, 1].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.show()


#NOW IT COMES THE MACHINE LEARNING PART

X = df.drop('sale_price', axis=1)
y = df['sale_price']

print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)
print("\nFirst 5 rows of X:")
print(X.head())
print("\nFirst 5 rows of y:")
print(y.head())

#processing the df
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Identify categorical and numerical columns
categorical_cols = ['borough_y', 'bldgclass']
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# Exclude categorical columns from numerical_cols if they were accidentally included
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols, index=X.index)

# Combine the processed features
X_processed = pd.concat([X_scaled_df, X_encoded_df], axis=1)

print("\nShape of X_processed:", X_processed.shape)
print("\nFirst 5 rows of X_processed:")
print(X_processed.head())

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

#train a MLPRegressor
from sklearn.neural_network import MLPRegressor

# Initialize the MLP Regressor model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Train the model
mlp_model.fit(X_train, y_train)

print("MLPRegressor model trained successfully.")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on the test set
y_pred = mlp_model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")

#train a random forest
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=60)

# Train the model
rf_model.fit(X_train, y_train)

print("Random Forest Regressor model trained successfully.")

#let's do some features engeneering to try to improve the model
  
# Create area_per_unit
# Handle division by zero: replace inf with NaN, then fill NaN with 0
X['area_per_unit'] = X['bldgarea'] / X['unitstotal']
X['area_per_unit'] = X['area_per_unit'].replace([np.inf, -np.inf], np.nan)
X['area_per_unit'] = X['area_per_unit'].fillna(0)

# Create lot_to_bldg_ratio
# Handle division by zero: replace inf with NaN, then fill NaN with 0
X['lot_to_bldg_ratio'] = X['lotarea'] / X['bldgarea']
X['lot_to_bldg_ratio'] = X['lot_to_bldg_ratio'].replace([np.inf, -np.inf], np.nan)
X['lot_to_bldg_ratio'] = X['lot_to_bldg_ratio'].fillna(0)

# Create has_commercial binary feature
X['has_commercial'] = (X['comarea'] > 0).astype(int)

print("Shape of X after adding new features:", X.shape)
print("\nFirst 5 rows of X with new features:")
print(X.head())

#now I will train again random forest with the new dataset: I added 3 features and replaced the bldgclass column with several columns referring to each building class
#now instead of having one single column with discrete values, we'll have several columns with binomial values (1 or 0)

# 1. Identify categorical and numerical columns in the updated X DataFrame
categorical_cols = ['borough_y', 'bldgclass']
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# Exclude categorical columns from numerical_cols if they were accidentally included
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

print("Categorical Columns (new):", categorical_cols)
print("Numerical Columns (new):", numerical_cols)

# 2. Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded_new = encoder.fit_transform(X[categorical_cols])
X_encoded_df_new = pd.DataFrame(X_encoded_new, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

# 3. Apply Standard Scaling to numerical features
scaler = StandardScaler()
X_scaled_new = scaler.fit_transform(X[numerical_cols])
X_scaled_df_new = pd.DataFrame(X_scaled_new, columns=numerical_cols, index=X.index)

# 4. Concatenate the encoded categorical features and scaled numerical features
X_processed_new = pd.concat([X_scaled_df_new, X_encoded_df_new], axis=1)

print("\nShape of X_processed_new:", X_processed_new.shape)
print("\nFirst 5 rows of X_processed_new:")
print(X_processed_new.head())

# 5. Split X_processed_new and y into training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_processed_new, y, test_size=0.2, random_state=42)

print("\nShape of X_train_new:", X_train_new.shape)
print("Shape of X_test_new:", X_test_new.shape)
print("Shape of y_train_new:", y_train_new.shape)
print("Shape of y_test_new:", y_test_new.shape)

# 6. Initialize a RandomForestRegressor model
rf_model_new = RandomForestRegressor(random_state=60)

# 7. Train this new Random Forest model
rf_model_new.fit(X_train_new, y_train_new)

print("\nRandom Forest Regressor model trained successfully with new features.")

  # Make predictions on the test set using the new model
y_pred_rf_new = rf_model_new.predict(X_test_new)

# Calculate regression metrics
mae_rf_new = mean_absolute_error(y_test_new, y_pred_rf_new)
mse_rf_new = mean_squared_error(y_test_new, y_pred_rf_new)
rmse_rf_new = np.sqrt(mse_rf_new)
r2_rf_new = r2_score(y_test_new, y_pred_rf_new)

# Print the metrics
print(f"Random Forest Model Performance with New Features:")
print(f"Mean Absolute Error (MAE): {mae_rf_new:,.2f}")
print(f"Mean Squared Error (MSE): {mse_rf_new:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf_new:,.2f}")
print(f"R-squared (R2): {r2_rf_new:.4f}")

#the result metrics still sucks, let's do some tuning of the hyperparameters.
# let's start to perform some tests to see which combination is the best one
#let's re-do the whole process from scratch starting from the original dataframe

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Re-define X and y from the original df and apply feature engineering
# Assuming df and y are still available from previous execution. If not, this needs to be reloaded.
# For this fix, I will assume df and y are available.

# Create area_per_unit
X['area_per_unit'] = X['bldgarea'] / X['unitstotal']
X['area_per_unit'] = X['area_per_unit'].replace([np.inf, -np.inf], np.nan)
X['area_per_unit'] = X['area_per_unit'].fillna(0)

# Create lot_to_bldg_ratio
X['lot_to_bldg_ratio'] = X['lotarea'] / X['bldgarea']
X['lot_to_bldg_ratio'] = X['lot_to_bldg_ratio'].replace([np.inf, -np.inf], np.nan)
X['lot_to_bldg_ratio'] = X['lot_to_bldg_ratio'].fillna(0)

# Create has_commercial binary feature
X['has_commercial'] = (X['comarea'] > 0).astype(int)

# 1. Identify categorical and numerical columns in the updated X DataFrame
categorical_cols = ['borough_y', 'bldgclass']
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

# Exclude categorical columns from numerical_cols if they were accidentally included
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

# 2. Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded_new = encoder.fit_transform(X[categorical_cols])
X_encoded_df_new = pd.DataFrame(X_encoded_new, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

# 3. Apply Standard Scaling to numerical features
scaler = StandardScaler()
X_scaled_new = scaler.fit_transform(X[numerical_cols])
X_scaled_df_new = pd.DataFrame(X_scaled_new, columns=numerical_cols, index=X.index)

# 4. Concatenate the encoded categorical features and scaled numerical features
X_processed_new = pd.concat([X_scaled_df_new, X_encoded_df_new], axis=1)

# 5. Split X_processed_new and y into training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_processed_new, y, test_size=0.2, random_state=42)

# --- Start Hyperparameter Tuning ---

# Define a parameter distribution dictionary
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

# Instantiate a RandomForestRegressor model
rf = RandomForestRegressor(random_state=60)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=10,
                                   cv=3,
                                   scoring='neg_mean_squared_error',
                                   random_state=42,
                                   n_jobs=-1,
                                   verbose=2)

# Fit random_search to the training data
print("Starting RandomizedSearchCV...")
random_search.fit(X_train_new, y_train_new)
print("RandomizedSearchCV completed.")

# Print the best parameters found
print("\nBest parameters found:")
print(random_search.best_params_)

# Get the best parameters from RandomizedSearchCV
best_params = random_search.best_params_
print(f"Best hyperparameters: {best_params}")

# Initialize a new RandomForestRegressor model with the best parameters
rf_model_tuned = RandomForestRegressor(random_state=60, **best_params)

# Train the tuned model
rf_model_tuned.fit(X_train_new, y_train_new)

print("Tuned Random Forest Regressor model trained successfully.")

# Make predictions on the test set using the tuned model
y_pred_rf_tuned = rf_model_tuned.predict(X_test_new)

# Calculate regression metrics for the tuned model
mae_rf_tuned = mean_absolute_error(y_test_new, y_pred_rf_tuned)
mse_rf_tuned = mean_squared_error(y_test_new, y_pred_rf_tuned)
rmse_rf_tuned = np.sqrt(mse_rf_tuned)
r2_rf_tuned = r2_score(y_test_new, y_pred_rf_tuned)

# Print the metrics for the tuned model
print(f"\nTuned Random Forest Model Performance:")
print(f"Mean Absolute Error (MAE): {mae_rf_tuned:,.2f}")
print(f"Mean Squared Error (MSE): {mse_rf_tuned:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf_tuned:,.2f}")
print(f"R-squared (R2): {r2_rf_tuned:.4f}")

#still no improvement, let's try to drop the irrilevant columns in the dataset
columns_to_keep = ['borough_x', 'block', 'sale_price', 'yearbuilt', 'lotarea', 'bldgarea', 'resarea', 'unitstotal', 'bldgclass', 'building_age']
df_subset = df[columns_to_keep].copy()
display(df_subset.head())

df_pulito=df_subset
print("First 5 rows of df_pulito:")
display(df_pulito.head())

print("\nInfo of df_pulito:")
df_pulito.info()
  
# Select only numerical columns from df_pulito
numerical_df_pulito = df_pulito.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix_pulito = numerical_df_pulito.corr()

# Extract correlations with 'sale_price'
sale_price_correlations_pulito = correlation_matrix_pulito[['sale_price']].sort_values(by='sale_price', ascending=False)

# Create a heatmap of the correlations with 'sale_price'
plt.figure(figsize=(8, 8))
sns.heatmap(sale_price_correlations_pulito, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix with Sale Price in df_pulito')
plt.show()    #correlation matrix doesn't seem to get any better from the previous one

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Plot for 'building_age' vs 'sale_price'
sns.regplot(x='building_age', y='sale_price', data=df_pulito, ax=axes[0], lowess=True, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='Non-linear Trend')
axes[0].set_title('Sale Price vs. Building Age (Non-Linear Trend)')
axes[0].set_xlabel('Building Age')
axes[0].set_ylabel('Sale Price')
axes[0].ticklabel_format(style='plain', axis='y')
axes[0].legend()

# Plot for 'bldgarea' vs 'sale_price'
sns.regplot(x='bldgarea', y='sale_price', data=df_pulito, ax=axes[1], lowess=True, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='Non-linear Trend')
axes[1].set_title('Sale Price vs. Building Area (Non-Linear Trend)')
axes[1].set_xlabel('Building Area')
axes[1].set_ylabel('Sale Price')
axes[1].ticklabel_format(style='plain', axis='y')
axes[1].legend()

# Plot for 'lotarea' vs 'sale_price'
sns.regplot(x='lotarea', y='sale_price', data=df_pulito, ax=axes[2], lowess=True, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, label='Non-linear Trend')
axes[2].set_title('Sale Price vs. Lot Area (Non-Linear Trend)')
axes[2].set_xlabel('Lot Area')
axes[2].set_ylabel('Sale Price')
axes[2].ticklabel_format(style='plain', axis='y')
axes[2].legend()

plt.tight_layout()
plt.show()  #also the scatterplot doesn't seem any better

X = df_pulito.drop('sale_price', axis=1)
y = df_pulito['sale_price']

print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)
print("\nFirst 5 rows of X:")
print(X.head())
print("\nFirst 5 rows of y:")
print(y.head())

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols, index=X.index)

# Combine the processed features
X_processed = pd.concat([X_scaled_df, X_encoded_df], axis=1)

print("\nShape of X_processed:", X_processed.shape)
print("\nFirst 5 rows of X_processed:")
print(X_processed.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Make predictions on the test set
y_pred_rf_original_features = rf_model_original_features.predict(X_test)

# Calculate regression metrics
mae_rf_original = mean_absolute_error(y_test, y_pred_rf_original_features)
mse_rf_original = mean_squared_error(y_test, y_pred_rf_original_features)
rmse_rf_original = np.sqrt(mse_rf_original)
r2_rf_original = r2_score(y_test, y_pred_rf_original_features)

# Print the metrics
print(f"Random Forest Model Performance (Original Features):")
print(f"Mean Absolute Error (MAE): {mae_rf_original:,.2f}")
print(f"Mean Squared Error (MSE): {mse_rf_original:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf_original:,.2f}")
print(f"R-squared (R2): {r2_rf_original:.4f}")  #and also the final metrics are not any better
