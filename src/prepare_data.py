import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_fer2013(input_file='fer2013.csv', output_dir='data'):
    """
    Prepare FER2013 dataset by splitting into train/val/test sets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['emotion']
    )
    
    # Further split train into train and validation
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_df['emotion']
    )
    
    # Save the splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Dataset split and saved to {output_dir}:")
    print(f"- Training samples: {len(train_df)}")
    print(f"- Validation samples: {len(val_df)}")
    print(f"- Test samples: {len(test_df)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare FER2013 dataset')
    parser.add_argument('--input', type=str, default='fer2013.csv',
                        help='Path to the fer2013.csv file')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to save the processed data')
    args = parser.parse_args()
    
    prepare_fer2013(args.input, args.output_dir)
