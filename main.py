import preproc
import pandas as pd
import matplotlib.pyplot as plt

# main.py


def main():
    # Load the csv data
    df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
    
    df = preproc.label_encode_target(df)
    df = preproc.one_hot_encode(df)
    plt.boxplot(df)
    #print amouunt of lines in dataset

    df = preproc.handle_outliers(df)
    df.to_csv('smoking_driking_dataset_Ver01_encoded.csv', index=False)
    #check df after outlier removal with pyplot and set maximum y value to 6000 for plt.show
    plt.boxplot(df)
 
    # Split the dataset
    x_train, x_test, y_train, y_test = preproc.split_dataset(df)

    # Min-Max scaling
    x_train, x_test = preproc.min_max_scale(x_train, x_test)

    # RFE
    x_train, x_test = preproc.perform_rfe(x_train, x_test, y_train, 15)

    # save x_test and train as csv
    pd.DataFrame(x_train).to_csv('x_train.csv', index=False)
    pd.DataFrame(x_test).to_csv('x_test.csv', index=False)


    # Outlier handling
  

    #save dataset to file
    
    
    

if __name__ == "__main__":
    main()