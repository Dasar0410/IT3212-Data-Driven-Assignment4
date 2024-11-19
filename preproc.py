import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

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

    return df_encoded



def split_dataset(df):
    x = df.drop('DRK_YN','Smoking_Never','Smoking_Former','Smoking_Current', axis=1)
    y = df['DRK_YN','Smoking_Never','Smoking_Former','Smoking_Current']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def target_encode(x_train, x_test, y_train, cat_cols):
    pass

def handle_outliers(df, outlier_columns, iqr_multiplier):
    pass

def perform_rfe(X_train, y_train, n_features_to_select):
    pass

def perform_lda(X_train_selected, y_train, n_components):
    pass

def min_max_scale(X_train_selected, X_test_selected):
    pass

def perform_pca(X_train_scaled, X_test_scaled, n_components):
    pass