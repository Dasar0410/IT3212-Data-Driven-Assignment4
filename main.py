import preproc
import pandas as pd

# main.py


def main():
    # Load the csv data
    df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
    
    df = preproc.one_hot_encode(df)
    df = preproc.label_encode_target(df)

    #save dataset to file
    df.to_csv('smoking_driking_dataset_Ver01_encoded.csv', index=False)
    
    

if __name__ == "__main__":
    main()