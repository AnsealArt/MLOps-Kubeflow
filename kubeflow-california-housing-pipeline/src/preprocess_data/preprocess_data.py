import numpy as np
import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split

def _preprocess_data(output_path, run_date, test_size=0.2):

    # Load data from sklearn datasets
    X, y = datasets.fetch_california_housing(return_X_y=True)

    # Split data for train and test sets using given test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Save split results
    np.save('{}{}/x_train.npy'.format(output_path, run_date), X_train)
    np.save('{}{}/x_test.npy'.format(output_path, run_date), X_test)
    np.save('{}{}/y_train.npy'.format(output_path, run_date), y_train)
    np.save('{}{}/y_test.npy'.format(output_path, run_date), y_test)
    

if __name__ == '__main__':
    
    # Receive arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--output_path', type=dir_path)
    parser.add_argument('--run_date')

    # If argument has been passed validate it
    args, leftovers = parser.parse_known_args()
    if args.test_size is not None:
        assert args.test_size > 0 and args.test_size < 1, "Test Size must be a value between 0 and 1, for example 0.3"

    # Run data preprocessing based on if test size has been provided
    if args.test_size is None:
        _preprocess_data(args.run_date, args.output_path)
    else:
        _preprocess_data(args.run_date, args.output_path, args.test_size)