import preproc
import pandas as pd

# main.py


def main():
    # Load the csv data
    df = pd.read_csv('smoking_driking_dataset_Ver01.csv')
    
    df = preproc.process_data(data)
    
    print("Processed Data:", processed_data)

if __name__ == "__main__":
    main()