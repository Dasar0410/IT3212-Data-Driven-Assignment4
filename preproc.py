import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def label_encode_target(df):
    # Convert 'DRK_YN' to 0 or 1
    df['DRK_YN'] = df['DRK_YN'].map({'N': 0, 'Y': 1})

    label_encoder = LabelEncoder()
    
    label_columns = ['hear_left', 'hear_right', 'urine_protein']

    for col in label_columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df

def one_hot_encode(df):
    df_encoded = pd.get_dummies(df, columns=['sex', 'SMK_stat_type_cd']).astype(int)

    # Rename columns to more fitting names
    df_encoded.rename(columns={
        'SMK_stat_type_cd_1.0': 'Smoking_Never',
        'SMK_stat_type_cd_2.0': 'Smoking_Former',
        'SMK_stat_type_cd_3.0': 'Smoking_Current'
    }, inplace=True)
    # return as pd dataframe
    df_encoded = pd.DataFrame(df_encoded)
    return df_encoded

def split_dataset(df):
    x = df.drop(['DRK_YN', 'Smoking_Never', 'Smoking_Former', 'Smoking_Current'], axis=1)
    y = df['DRK_YN']
    # Handle outliers before splitting the dataset but after dividing the target and features
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test

def handle_outliers(df):
    # Identify all numerical columns for outlier detection
    outlier_columns = [
        'height', 'weight', 'waistline', 
        'SBP', 'DBP', 'BLDS', 
        'tot_chole', 'HDL_chole', 'LDL_chole', 
        'triglyceride', 'hemoglobin', 
        'serum_creatinine', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP'
    ]
    # Set the IQR multiplier (adjust for stricter outlier detection)
    iqr_multiplier = 3
    
    for col in outlier_columns:
        lower_bound, upper_bound = calculate_iqr_bounds(df, col, iqr_multiplier)
        outlier_count = count_outliers(df, col, lower_bound, upper_bound)
        
        print(f"Processing '{col}':")
        print(f"  - Lower Bound: {lower_bound}")
        print(f"  - Upper Bound: {upper_bound}")
        print(f"  - Outliers Detected: {outlier_count}")
        
        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

def calculate_iqr_bounds(df, column, multiplier):
    """Calculate the IQR bounds for a given column with a specified multiplier."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return lower_bound, upper_bound

def count_outliers(df, column, lower_bound, upper_bound):
    """Count the number of outliers in a column based on IQR bounds."""
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)
    
def min_max_scale(X_train_selected, X_test_selected):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Convert back to DataFrame to retain column names and indices
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_selected.columns, index=X_train_selected.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_selected.columns, index=X_test_selected.index)
    
    # Debugging: Print the range of scaled values
    print("Training Data Range After Scaling:")
    print("Min:\n", X_train_scaled.min())
    print("Max:\n", X_train_scaled.max())

    return X_train_scaled, X_test_scaled

def perform_rfe(X_train, X_test, y_train, n_features_to_select):
    #RFE AND LDA
    # Create an estimator to be used by RFE
    estimator = LogisticRegression(max_iter=2000)
    rfe = RFE(estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    # Select the features that RFE gets
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    return X_train_selected, X_test_selected

def perform_lda(X_train_selected, y_train, n_components):
    pass

def perform_pca(X_train_scaled, X_test_scaled, n_components):
    pass