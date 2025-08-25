import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


from src.exception import CustomException
from src.logger import logging

import os

from src.utils import save_object


# Add missing-indicator columns for numerical vars
def replace_missing_cats(X):
    X = X.copy()
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        X[col] = X[col].fillna("Missing")
    return X

# handle missing numerical values
def impute_numerical(X):
    X = X.copy()
    #X = X.dropna(axis=1, how="all")
    num_cols = X.select_dtypes(exclude="object").columns
    for col in num_cols:
        median_value=X[col].median()
        if X[col].isnull().sum() > 0:
            X[col].fillna(median_value,inplace=True)
    return X

# Handle date features (convert to "age")
def transform_dates(X):
    X = X.copy()
    for col in ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]:
        if col in X.columns and "YrSold" in X.columns:
            X[col] = X["YrSold"] - X[col]
    return X

# Log-transform specific numerical features
def log_transform(X):
    X = X.copy()
    for col in ["LotFrontage", "LotArea", "1stFlrSF", "GrLivArea"]:
        if col in X.columns:
            X[col] = np.log(X[col].clip(lower=1))  # avoid log(0)
    return X

# Handle rare categories (<1% frequency → "Rare_var")
def rare_categories(X, threshold=0.01):
    X = X.copy()
    cat_cols = X.select_dtypes(include="object").columns
    for col in cat_cols:
        freqs = X[col].value_counts(normalize=True)
        rare_labels = freqs[freqs < threshold].index
        X[col] = np.where(X[col].isin(rare_labels), "Rare_var", X[col])
    return X

def make_target_rank_encoder(X):
    X = X.copy()
    categorical_features = X.select_dtypes(include="object").columns
    for feature in categorical_features:
        labels_ordered=X.groupby([feature])['LotFrontage'].mean().sort_values().index
        labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
        X[feature]=X[feature].map(labels_ordered)
    return X

def Scaling(X):
    X = X.copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)  # returns numpy array
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:

            preprocess_pipeline = Pipeline(steps=[
                ("replace_missing_cats", FunctionTransformer(replace_missing_cats)),
                ("impute_numerical", FunctionTransformer(impute_numerical)),
                ("transform_dates", FunctionTransformer(transform_dates)),
                ("rare_categories", FunctionTransformer(rare_categories)),
                ("make_target_rank_encoder", FunctionTransformer(make_target_rank_encoder)),
                ("log_transform", FunctionTransformer(log_transform)),
                ("scaler", FunctionTransformer(Scaling))
            ])

            return preprocess_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="SalePrice"

            input_feature_train_df=train_df.drop(columns=[target_column_name,"Id"],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name,"Id"],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_df=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_train_arr=input_feature_train_df.values

            input_feature_test_df=preprocessing_obj.transform(input_feature_test_df)
            input_feature_test_arr = input_feature_test_df.values


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

