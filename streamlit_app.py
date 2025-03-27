import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from streamlit_lottie import st_lottie
import json
import requests

# Page configuration
st.set_page_config(
    page_title="AI Code Assistant for Data Science",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #D5F5E3;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #D6EAF8;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_success = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_t26law.json")

# Sidebar
with st.sidebar:
    st.title("🎛️ Settings & Info")
    st.markdown("---")
    st.markdown("### About this tool")
    st.info(
        """
        This AI-powered assistant helps you with data science tasks by generating Python code based on your dataset and requirements.
        
        **Features:**
        - Data cleaning suggestions
        - Exploratory data analysis (EDA)
        - Model building code
        - Visualization recommendations
        
        Made with ❤️ by Sai Rupa Jhade
        """
    )
    
    st.markdown("---")
    st.markdown("### Sample Tasks to Try")
    st.markdown(
        """
        - "Clean this dataset and handle missing values"
        - "Create visualizations to explore relationships"
        - "Build a classification model to predict [column]"
        - "Perform feature engineering on numeric columns"
        """
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="main-header">🤖 AI Code Assistant for Data Science</p>', unsafe_allow_html=True)
    st.markdown(
        """
        Welcome to your interactive **AI-powered code assistant**! 🚀 
        
        Upload your dataset, describe your task, and let the assistant suggest code, cleaning tips, and analysis ideas.
        """
    )

with col2:
    st_lottie(lottie_coding, height=200, key="coding")

# File upload section
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Display dataset preview
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! Found {df.shape[0]} rows and {df.shape[1]} columns.")
        
        with st.expander("📊 Preview Your Dataset"):
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Dataset Info")
                buffer = df.dtypes.to_frame().reset_index()
                buffer.columns = ["Column", "Data Type"]
                st.table(buffer)
            
            with col2:
                st.markdown("#### 🔍 Missing Values")
                missing_data = df.isnull().sum().to_frame().reset_index()
                missing_data.columns = ["Column", "Missing Values"]
                missing_data["Missing %"] = (missing_data["Missing Values"] / len(df) * 100).round(2)
                st.table(missing_data)
    
    except Exception as e:
        st.error(f"Error reading the file: {e}")

# Task input
st.markdown('<p class="sub-header">✍️ What would you like to do with this data?</p>', unsafe_allow_html=True)
user_task = st.text_area(
    "Describe your data science task in detail",
    placeholder="e.g., Clean this dataset and handle missing values, then build a model to predict customer churn based on the 'Churn' column.",
    height=100
)

# Task type selection for more targeted responses
task_type = st.selectbox(
    "Select the primary task type:",
    ["Data Cleaning & Preprocessing", "Exploratory Data Analysis", "Feature Engineering", 
     "Model Building", "Visualization", "Custom Task"]
)

# Response area
if st.button("✨ Generate Code Suggestions", use_container_width=True):
    if uploaded_file and user_task:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            
            # Update status text based on progress
            if i < 20:
                status_text.text("🔍 Analyzing your dataset...")
            elif i < 40:
                status_text.text("🧠 Thinking about your task...")
            elif i < 70:
                status_text.text("✍️ Generating code suggestions...")
            elif i < 90:
                status_text.text("🔧 Optimizing recommendations...")
            else:
                status_text.text("🎁 Finalizing results...")
                
            time.sleep(0.05)
            
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success animation
        st_lottie(lottie_success, height=150, key="success")
        
        # Generate mock response based on task type
        if task_type == "Data Cleaning & Preprocessing":
            code_suggestion = """
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\\nMissing values per column:")
print(df.isnull().sum())

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
processed_data = preprocessor.fit_transform(df)

print("\\nPreprocessing complete!")
print(f"Processed data shape: {processed_data.shape}")

# Save preprocessor for later use
import joblib
joblib.dump(preprocessor, 'preprocessor.pkl')
"""
        elif task_type == "Exploratory Data Analysis":
            code_suggestion = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Set up the matplotlib figure size
plt.figure(figsize=(15, 10))

# 1. Summary statistics
print("Summary Statistics:")
print(df.describe().T)

# 2. Correlation heatmap
plt.figure(figsize=(12, 10))
correlation = df.select_dtypes(include=['int64', 'float64']).corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# 3. Distribution of numeric features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
n_features = len(numeric_features)
n_rows = (n_features + 2) // 3  # Ceiling division

fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, feature in enumerate(numeric_features):
    if i < len(axes):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.savefig('numeric_distributions.png')
plt.show()

# 4. Categorical feature analysis
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    value_counts = df[feature].value_counts().sort_values(ascending=False)
    
    # Plot top 10 categories if there are many
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
        title_suffix = " (Top 10)"
    else:
        title_suffix = ""
        
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Count of {feature}{title_suffix}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{feature}_counts.png')
    plt.show()

# 5. Pairplot for key numeric features
if len(numeric_features) > 1:
    # Select a subset of features if there are too many
    plot_features = numeric_features[:5] if len(numeric_features) > 5 else numeric_features
    sns.pairplot(df[plot_features])
    plt.suptitle('Pairplot of Key Numeric Features', y=1.02)
    plt.savefig('pairplot.png')
    plt.show()

# 6. Missing values visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.show()
"""
        elif task_type == "Model Building":
            code_suggestion = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Assume 'target_column' is your target variable - replace with actual column name
target_column = 'target_column'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and compare multiple models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.show()

# Compare model performance
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for i, (name, accuracy) in enumerate(results.items()):
    plt.text(i, accuracy + 0.01, f'{accuracy:.4f}', ha='center')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Get the best model
best_model_name = max(results, key=results.get)
print(f"\\nBest model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

# Fine-tune the best model with GridSearchCV
if best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
else:  # Logistic Regression
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__solver': ['liblinear', 'saga']
    }

# Create pipeline with the best model
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])
])

# Perform grid search
grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print best parameters
print("\\nBest parameters:")
print(grid_search.best_params_)

# Evaluate the tuned model
y_pred_tuned = grid_search.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"\\nTuned {best_model_name} Results:")
print(f"Accuracy: {accuracy_tuned:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))

# Save the final model
import joblib
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
print("\\nFinal model saved as 'best_model.pkl'")
"""
        else:
            code_suggestion = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\\nFirst few rows:")
print(df.head())
print("\\nColumn information:")
print(df.info())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Fill missing values
# For numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# For categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Exploratory Data Analysis
# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Prepare data for modeling
# Assuming 'target' is your target column - replace with actual column name
X = df.drop('target', axis=1)
y = df['target']

# Handle categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\\nModel Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.show()

print("\\nAnalysis complete!")
"""
        
        # Display the code suggestion
        st.markdown('<div class="success-box">✅ Code generated successfully!</div>', unsafe_allow_html=True)
        st.code(code_suggestion, language="python")
        
        # Add download button for the code
        st.download_button(
            label="📥 Download Python Code",
            data=code_suggestion,
            file_name="ai_generated_code.py",
            mime="text/plain",
        )
        
        # Add explanation
        st.markdown("### 📝 Explanation")
        st.markdown(
            """
            This code provides a comprehensive solution for your task. Here's what it does:
            
            1. **Data Loading & Inspection**: Loads your CSV file and examines its structure
            2. **Data Cleaning**: Handles missing values and prepares the data
            3. **Analysis/Modeling**: Implements the requested task with best practices
            4. **Visualization**: Creates informative plots to help understand the data and results
            5. **Evaluation**: Provides metrics to assess the quality of the results
            
            You can copy this code to your Python environment or download it as a file.
            """
        )
        
        # Add tips
        st.markdown("### 💡 Tips for Best Results")
        st.info(
            """
            - Make sure your column names don't contain spaces or special characters
            - Replace 'your_dataset.csv' with your actual file path
            - Adjust the target column name to match your dataset
            - Consider feature engineering to improve model performance
            - Try different models and parameters for better results
            """
        )
        
    else:
        st.warning("Please upload a CSV file and describe your task first.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Made with ❤️ by Sai Rupa Jhade | Data Science Portfolio
    </div>
    """, 
    unsafe_allow_html=True
)

