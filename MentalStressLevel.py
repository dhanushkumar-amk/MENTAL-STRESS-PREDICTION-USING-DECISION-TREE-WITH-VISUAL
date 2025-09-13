# Mental Stress Level Prediction using Decision Tree - Kaggle Dataset
# Complete Google Colab Implementation Guide

# ============================================================================
# STEP 1: MOUNT GOOGLE DRIVE AND SET UP KAGGLE
# ============================================================================

# Mount Google Drive to access your files
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install kaggle
!pip install plotly

# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import os
import json
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# ============================================================================
# STEP 2: SET UP KAGGLE API CREDENTIALS
# ============================================================================

# Copy kaggle.json from your Google Drive to the correct location
!mkdir -p ~/.kaggle
!cp "/content/drive/My Drive/Colab Notebooks/kaggle.json" ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle credentials set up successfully!")

# Verify Kaggle API is working
!kaggle --version

# ============================================================================
# STEP 3: LOAD YOUR DATASET FROM GOOGLE DRIVE
# ============================================================================

# Update these paths according to your actual file names in Google Drive
# Replace 'your_dataset.csv' with your actual dataset filename

# List files in your Colab Notebooks directory
print("📁 Files in your Colab Notebooks directory:")
drive_path = "/content/drive/My Drive/Colab Notebooks/"
files = os.listdir(drive_path)
for file in files:
    print(f"  - {file}")

# Load your dataset - Update the filename below
# Common mental health/stress dataset filenames:
dataset_files = [f for f in files if f.endswith('.csv')]
print(f"\n📊 Found CSV files: {dataset_files}")

# If you have multiple CSV files, specify which one to use
if dataset_files:
    dataset_filename = dataset_files[0]  # Uses the first CSV file found
    print(f"📁 Loading dataset: {dataset_filename}")
    
    df = pd.read_csv(drive_path + dataset_filename)
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
else:
    print("❌ No CSV files found. Please check your file names.")
    # Create sample data if no dataset is found
    print("📝 Creating sample dataset for demonstration...")
    
    # Sample dataset creation (remove this if you have real data)
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(4, 12),
        'work_hours': np.random.normal(8, 2, n_samples).clip(4, 14),
        'exercise_hours': np.random.exponential(1, n_samples).clip(0, 4),
        'social_media_hours': np.random.normal(3, 2, n_samples).clip(0, 10),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'job_satisfaction': np.random.randint(1, 11, n_samples),
        'relationship_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'chronic_illness': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    })
    
    # Create stress level based on features
    stress_factors = (
        (df['sleep_hours'] < 6).astype(int) * 2 +
        (df['work_hours'] > 9).astype(int) * 2 +
        (df['exercise_hours'] < 0.5).astype(int) * 1 +
        (df['social_media_hours'] > 6).astype(int) * 1 +
        (df['job_satisfaction'] < 5).astype(int) * 2 +
        df['chronic_illness'] * 2 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    df['stress_level'] = pd.cut(stress_factors, 
                               bins=[-np.inf, 1, 3, 5, np.inf], 
                               labels=['Low', 'Moderate', 'High', 'Very High'])

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# ============================================================================

print("📊 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nBasic statistics:")
print(df.describe())

# ============================================================================
# STEP 5: DATA PREPROCESSING
# ============================================================================

# Check if 'stress_level' column exists (adjust column name if needed)
target_columns = ['stress_level', 'stress', 'mental_health', 'anxiety_level', 'depression_level']
target_col = None

for col in target_columns:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("❌ No target column found. Please specify the correct column name for stress/mental health.")
    # For demonstration, we'll assume the last column is the target
    target_col = df.columns[-1]
    print(f"📌 Using '{target_col}' as target column")

print(f"🎯 Target variable: {target_col}")
print(f"Target distribution:\n{df[target_col].value_counts()}")

# Prepare features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Handle categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"📝 Categorical columns: {list(categorical_columns)}")

# Label encode categorical variables
label_encoders = {}
X_processed = X.copy()

for col in categorical_columns:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable if it's categorical
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    target_classes = target_encoder.classes_
    print(f"🏷️ Target classes: {target_classes}")
else:
    y_encoded = y
    target_classes = sorted(y.unique())

print(f"✅ Data preprocessing completed!")

# ============================================================================
# STEP 6: INTERACTIVE VISUALIZATIONS
# ============================================================================

# Create comprehensive visualizations
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Target Distribution', 'Feature Correlation Heatmap', 
                   'Feature Importance (Preview)', 'Stress by Age Groups',
                   'Feature Distributions', 'Missing Values'),
    specs=[[{"type": "bar"}, {"type": "heatmap"}],
           [{"type": "bar"}, {"type": "box"}],
           [{"type": "histogram"}, {"type": "bar"}]]
)

# 1. Target Distribution
target_counts = df[target_col].value_counts()
fig.add_trace(
    go.Bar(x=target_counts.index, y=target_counts.values, name="Stress Levels",
           marker_color='lightblue'),
    row=1, col=1
)

# 2. Correlation heatmap for numeric columns
numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr_matrix = X_processed[numeric_cols].corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                  colorscale='RdBu', zmid=0),
        row=1, col=2
    )

fig.update_layout(height=1200, showlegend=False, title_text="Mental Stress Analysis Dashboard")
fig.show()

# Additional individual plots
plt.figure(figsize=(20, 15))

# Plot 1: Target distribution
plt.subplot(3, 4, 1)
df[target_col].value_counts().plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('Distribution of Stress Levels')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 2: Correlation heatmap
if len(numeric_cols) > 1:
    plt.subplot(3, 4, 2)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')

# Plot 3-6: Box plots for key numeric features
numeric_features = list(numeric_cols)[:4]  # First 4 numeric features
for i, feature in enumerate(numeric_features):
    plt.subplot(3, 4, i+3)
    if feature in X.columns:
        df.boxplot(column=feature, by=target_col, ax=plt.gca())
        plt.title(f'{feature} by Stress Level')
        plt.suptitle('')

# Plot 7: Feature distributions
plt.subplot(3, 4, 7)
if len(numeric_cols) > 0:
    X_processed[numeric_cols[0]].hist(bins=30, alpha=0.7, color='green')
    plt.title(f'Distribution of {numeric_cols[0]}')

# Plot 8: Missing values
plt.subplot(3, 4, 8)
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
if len(missing_counts) > 0:
    missing_counts.plot(kind='bar', color='red', alpha=0.7)
    plt.title('Missing Values Count')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Missing Values Status')

plt.tight_layout()
plt.show()

print("📈 Visualizations completed!")

# ============================================================================
# STEP 7: BUILD AND TRAIN DECISION TREE MODEL
# ============================================================================

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"🔄 Data split completed:")
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Train Decision Tree with hyperparameter tuning
print("🌳 Training Decision Tree model...")

# Grid search for best parameters
param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_classifier = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_
print(f"✅ Best parameters: {grid_search.best_params_}")

# Train the best model
best_dt.fit(X_train, y_train)

# Make predictions
y_pred = best_dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")

# Cross-validation score
cv_scores = cross_val_score(best_dt, X_processed, y_encoded, cv=5)
print(f"📊 Cross-validation scores: {cv_scores}")
print(f"📊 Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 8: MODEL EVALUATION AND VISUALIZATION
# ============================================================================

# Classification Report
print("\n📋 Classification Report:")
if y.dtype == 'object':
    print(classification_report(y_test, y_pred, target_names=target_classes))
else:
    print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(15, 12))

# Plot 1: Confusion Matrix
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_classes if y.dtype == 'object' else sorted(set(y_test)),
            yticklabels=target_classes if y.dtype == 'object' else sorted(set(y_test)))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 2: Feature Importance
plt.subplot(2, 3, 2)
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': best_dt.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')

# Plot 3: Decision Tree Visualization (simplified)
plt.subplot(2, 3, 3)
plot_tree(best_dt, max_depth=3, feature_names=X_processed.columns[:10], 
          class_names=target_classes if y.dtype == 'object' else None,
          filled=True, fontsize=8)
plt.title('Decision Tree (Depth=3)')

# Plot 4: Model Performance Comparison
plt.subplot(2, 3, 4)
models_comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Baseline (Random)'],
    'Accuracy': [accuracy, 1/len(target_classes)]
})
plt.bar(models_comparison['Model'], models_comparison['Accuracy'], 
        color=['lightgreen', 'lightcoral'])
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Plot 5: Cross-validation scores
plt.subplot(2, 3, 5)
plt.plot(range(1, len(cv_scores)+1), cv_scores, 'bo-')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot 6: Prediction distribution
plt.subplot(2, 3, 6)
pred_counts = pd.Series(y_pred).value_counts().sort_index()
actual_counts = pd.Series(y_test).value_counts().sort_index()

x_pos = np.arange(len(pred_counts))
plt.bar(x_pos - 0.2, actual_counts.values, 0.4, label='Actual', alpha=0.7)
plt.bar(x_pos + 0.2, pred_counts.values, 0.4, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Distribution')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.xticks(x_pos, actual_counts.index)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 9: INTERACTIVE FEATURE IMPORTANCE VISUALIZATION
# ============================================================================

# Interactive feature importance plot
fig_importance = px.bar(
    feature_importance.head(15), 
    x='importance', 
    y='feature',
    orientation='h',
    title='Feature Importance in Mental Stress Prediction',
    color='importance',
    color_continuous_scale='viridis'
)
fig_importance.update_layout(height=600)
fig_importance.show()

# ============================================================================
# STEP 10: PREDICTION FUNCTION AND SAMPLE PREDICTIONS
# ============================================================================

def predict_stress_level(model, feature_names, encoders, target_encoder, **kwargs):
    """
    Predict stress level for new data
    """
    # Create DataFrame with input features
    input_data = pd.DataFrame([kwargs])
    
    # Encode categorical variables
    for col in input_data.columns:
        if col in encoders:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except ValueError:
                print(f"Warning: Unknown category in {col}")
                input_data[col] = 0  # Default encoding
    
    # Ensure all required features are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0  # Default value
    
    # Select only the features used in training
    input_data = input_data[feature_names]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Decode prediction if necessary
    if hasattr(target_encoder, 'inverse_transform'):
        prediction_label = target_encoder.inverse_transform([prediction])[0]
    else:
        prediction_label = prediction
    
    return prediction_label, probability

# Example prediction
print("\n🔮 Example Stress Level Prediction:")
print("=" * 50)

# Create sample input based on your dataset features
sample_input = {}
for col in X_processed.columns:
    if col in categorical_columns:
        # Use mode for categorical
        sample_input[col] = df[col].mode()[0] if col in df.columns else 'Unknown'
    else:
        # Use mean for numeric
        sample_input[col] = df[col].mean() if col in df.columns else 0

print("Sample input features:")
for key, value in sample_input.items():
    if key in X.columns[:10]:  # Show first 10 features
        print(f"  {key}: {value}")

if y.dtype == 'object':
    prediction, probabilities = predict_stress_level(
        best_dt, X_processed.columns, label_encoders, target_encoder, **sample_input
    )
    print(f"\nPredicted Stress Level: {prediction}")
    print("Prediction Probabilities:")
    for i, class_name in enumerate(target_classes):
        print(f"  {class_name}: {probabilities[i]:.3f}")
else:
    prediction = best_dt.predict(pd.DataFrame([sample_input]))[0]
    print(f"\nPredicted Stress Level: {prediction}")

# ============================================================================
# STEP 11: SAVE MODEL AND RESULTS
# ============================================================================

print("\n💾 Saving model and results...")

# Save feature importance
feature_importance.to_csv('/content/drive/My Drive/Colab Notebooks/feature_importance.csv', index=False)

# Save model predictions
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})
results_df.to_csv('/content/drive/My Drive/Colab Notebooks/model_predictions.csv', index=False)

print("✅ Files saved to your Google Drive:")
print("  - feature_importance.csv")
print("  - model_predictions.csv")

print("\n🎉 Mental Stress Prediction Model Complete!")
print("=" * 60)
print(f"Final Model Accuracy: {accuracy:.4f}")
print(f"Model Type: Decision Tree")
print(f"Features Used: {len(X_processed.columns)}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print("=" * 60)
