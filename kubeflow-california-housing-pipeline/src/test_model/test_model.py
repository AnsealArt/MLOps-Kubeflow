import argparse
import joblib
import numpy as np

from sklearn.metrics import mean_squared_error


# Test model and output MSE metric
def test_model(x_test, y_test, model_path, output_path, run_date):
    x_test_data = np.load(x_test)
    y_test_data = np.load(y_test)

    model = joblib.load(model_path)

    y_pred = model.predict(x_test_data)

    err = mean_squared_error(y_test_data, y_pred)

    with open('{}{}/output.txt'.format(output_path, run_date), 'a') as file:
        file.write(str(err))


if __name__ == '__main__':
    
    # Parse received arguments with test data
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model_path')
    parser.add_argument('--output_path')
    parser.add_argument('--run_date')

    args = parser.parse_args()
    
    # Test created model
    test_model(args.x_test, args.y_test, args.model_path, args.output_path, args.run_date)
