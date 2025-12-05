import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc




# Synthetic Data Generation (Simulation)

def generate_stroke_data(n_samples=1000, random_seed=42):
    """
    Generates a synthetic dataset reflecting acute ischemic stroke parameters.
    Variables chosen based on common risk factors for Early Neurological Deterioration (END).
    """
    np.random.seed(random_seed)
    
    data = {
        'age': np.random.normal(68, 12, n_samples).astype(int),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'systolic_bp': np.random.normal(150, 20, n_samples),  # Hypertension is common
        'diastolic_bp': np.random.normal(90, 10, n_samples),
        'glucose_level': np.random.normal(140, 40, n_samples), # Stress hyperglycemia
        'admission_nihss': np.random.poisson(8, n_samples),   # Stroke severity score
        'atrial_fibrillation': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'prior_stroke': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to realistic medical ranges
    df['age'] = df['age'].clip(30, 100)
    df['systolic_bp'] = df['systolic_bp'].clip(90, 220)
    df['glucose_level'] = df['glucose_level'].clip(60, 400)
    df['admission_nihss'] = df['admission_nihss'].clip(0, 42)

    # Introduce some missing values (common in clinical data)
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, 'glucose_level'] = np.nan

    # Target Generation: Early Neurological Deterioration (END)
    # We simulate END based on higher NIHSS, Glucose, and BP (simple linear combination + noise)
    logit = (
        0.05 * df['age'] + 
        0.15 * df['admission_nihss'] + 
        0.01 * df['systolic_bp'] + 
        0.005 * df['glucose_level'] - 
        12 # Intercept adjustment
    )
    prob_end = 1 / (1 + np.exp(-logit))
    df['END_target'] = (np.random.random(n_samples) < prob_end).astype(int)
    
    return df





# 2. Feature Engineering

def engineer_features(df):
    """
    Creates clinically relevant features.
    """
    df_eng = df.copy()
    
    # 1. Mean Arterial Pressure (MAP)
    # MAP = (SBP + 2*DBP) / 3
    # Relevance: Cerebral perfusion pressure depends on MAP.
    df_eng['mean_arterial_pressure'] = (df_eng['systolic_bp'] + 2 * df_eng['diastolic_bp']) / 3
    
    # 2. Glucose-to-NIHSS Ratio (Hypothetical stress marker)
    # Filling NaN temporarily for calculation, though pipeline handles it later
    filled_glucose = df_eng['glucose_level'].fillna(df_eng['glucose_level'].median())
    df_eng['glucose_nihss_interaction'] = filled_glucose * df_eng['admission_nihss']
    
    return df_eng





# 3. Exploratory Data Analysis (EDA)

def plot_eda(df):
    """
    Generates visualizations for the dataset.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3)
    
    # Plot 1: Age Distribution by END Status
    end_positive = df[df['END_target'] == 1]['age']
    end_negative = df[df['END_target'] == 0]['age']
    axes[0, 0].hist(end_negative, alpha=0.6, label='No END', bins=20, color='skyblue')
    axes[0, 0].hist(end_positive, alpha=0.6, label='END Observed', bins=20, color='salmon')
    axes[0, 0].set_title('Age Distribution Stratified by END')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].legend()

    # Plot 2: Admission NIHSS Boxplot
    # Manual boxplot creation with matplotlib
    data_to_plot = [df[df['END_target']==0]['admission_nihss'], df[df['END_target']==1]['admission_nihss']]
    axes[0, 1].boxplot(data_to_plot, labels=['No END', 'END Observed'])
    axes[0, 1].set_title('Admission NIHSS Score vs END')
    axes[0, 1].set_ylabel('NIHSS Score')

    # Plot 3: Scatter of MAP vs Glucose
    scatter = axes[1, 0].scatter(
        df['mean_arterial_pressure'], 
        df['glucose_level'], 
        c=df['END_target'], 
        cmap='coolwarm', 
        alpha=0.6,
        edgecolors='w'
    )
    axes[1, 0].set_title('MAP vs Glucose Level (Color=END)')
    axes[1, 0].set_xlabel('Mean Arterial Pressure (mmHg)')
    axes[1, 0].set_ylabel('Glucose (mg/dL)')
    fig.colorbar(scatter, ax=axes[1, 0], label='END Target (1=Yes)')

    # Plot 4: Class Balance
    counts = df['END_target'].value_counts()
    axes[1, 1].bar(counts.index.astype(str), counts.values, color=['skyblue', 'salmon'])
    axes[1, 1].set_title('Target Class Balance')
    axes[1, 1].set_xlabel('END Status')
    axes[1, 1].set_ylabel('Count')

    plt.suptitle('Stroke Dataset Exploratory Analysis', fontsize=16)
    plt.show()





# 4. Pipeline Construction & Modeling

def run_pipeline():
    print("--- 1. Generating Synthetic Clinical Data ---")
    raw_df = generate_stroke_data()
    print(f"Dataset shape: {raw_df.shape}")
    print(raw_df.head())
    
    print("\n--- 2. Feature Engineering ---")
    df = engineer_features(raw_df)
    
    print("\n--- 3. Exploratory Visualization ---")
    print("Generating plots...")
    plot_eda(df)
    
    # Separation of features and target
    X = df.drop(columns=['END_target'])
    y = df['END_target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define Column Types
    numeric_features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 
                        'admission_nihss', 'mean_arterial_pressure', 'glucose_nihss_interaction']
    categorical_features = ['sex', 'atrial_fibrillation', 'prior_stroke']
    
    # Create Transformers
    # Impute numeric with median (robust to outliers like extreme glucose)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # One-hot encode categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine into Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Full Modeling Pipeline
    # Using 'class_weight="balanced"' to handle potential imbalance in END cases
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    print("\n--- 4. Training Model ---")
    clf.fit(X_train, y_train)
    
    print("\n--- 5. Evaluation ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Predicting Early Neurological Deterioration')
    plt.legend(loc="lower right")
    plt.show()
    
    # Feature Importance Analysis (Odds Ratios)
    # Extract feature names after one-hot encoding
    model = clf.named_steps['classifier']
    preprocessor_step = clf.named_steps['preprocessor']
    
    cat_names = preprocessor_step.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_names)
    
    coeffs = model.coef_[0]
    odds_ratios = np.exp(coeffs)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coeffs,
        'Odds_Ratio': odds_ratios
    }).sort_values(by='Odds_Ratio', ascending=False)
    
    print("\n--- Feature Importance (Odds Ratios) ---")
    print(importance_df)
    
if __name__ == "__main__":
    run_pipeline()
