import argparse
import joblib
import numpy as np

from datetime import datetime

from sklearn.linear_model import SGDRegressor


# Train model using SGDRegressor
def train_model(x_train, y_train, output_path, run_date):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)

    model = SGDRegressor(verbose=1)
    model.fit(x_train_data, y_train_data)

    joblib.dump(model, '{}{}/model.pkl'.format(output_path, run_date))


if __name__ == '__main__':

    # Parse received arguments with training data
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--output_path', type=dir_path)
    parser.add_argument('--run_date')

    args = parser.parse_args()
    
    # Train the regression model
    train_model(args.x_train, args.y_train, args.output_path, args.run_date)
