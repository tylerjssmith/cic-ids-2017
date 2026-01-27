"""
Train machine learning model for network intrusion detection system
"""
import pandas as pd
from datetime import datetime
from src.processing import (
    load_data, 
    clean_data, 
    prepare_labels_binary, 
    split_data
)


def main():
    print('='*70)
    print("Network Intrusion Detection System")
    print("Train Machine Learning Model")
    print(f"Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print('='*70)
    print('\n')

    # Load and prepare data
    df = load_data('raw','Wednesday-workingHours.pcap_ISCX.csv')
    df = clean_data(df)
    df = prepare_labels_binary(df, exclude_values=['Heartbleed'])
    df_train, df_test = split_data(df)

    # Save data splits
    df_train.to_parquet('data/intermediate/df_train.parquet')
    df_test.to_parquet('data/intermediate/df_test.parquet')

    print('='*70)
    print(f"End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print('='*70)
    print('\n')


if __name__ == "__main__":
    main()