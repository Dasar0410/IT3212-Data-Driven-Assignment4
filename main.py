import preproc
import pandas as pd

# main.py


def main():
    # Load the csv data
    df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
    
    df = preproc.label_encode_target(df)
    df = preproc.one_hot_encode(df)
    df = preproc.handle_outliers(df)
    df.to_csv('smoking_driking_dataset_Ver01_encoded.csv', index=False)
    x_train, x_test, y_train, y_test = preproc.split_dataset(df)
    #save train and test dataset to file
    x_train.to_csv('x_train.csv', index=False)
    x_test.to_csv('x_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # Outlier handling
  

    #save dataset to file
    
    
    

if __name__ == "__main__":
    main()