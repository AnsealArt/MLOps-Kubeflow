import json
import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def _download_data(args):

    # Get data split by attributes and target
    x, y = load_breast_cancer(return_X_y=True)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create JSON-like data structure
    data = {
        'x_train' : x_train.tolist() ,
        'y_train' : y_train.tolist() ,
        'x_test'  : x_test.tolist()  ,
        'y_test'  : y_test.tolist()
    }

    # Parse data to JSON format
    data_json = json.dumps(data)

    # Save JSON data to file ==> Artifact
    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)

# Define main as a script start
if __name__ == '__main__':

    # Create arguments
    # IN: null
    # OUT: data
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    
    args = parser.parse_args()

    # Place data into specified in arguments file, directory is in 'create if not exists' mode
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    # Run the script
    _download_data(args)