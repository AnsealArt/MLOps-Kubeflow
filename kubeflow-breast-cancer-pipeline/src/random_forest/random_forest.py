import json
import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def _random_forest(args):

    # Read JSON data
    with open(args.data) as data_file:
        data = json.load(data_file)
    
    # data variable is string type, so we need to reload it into JSON object
    data = json.loads(data)

    # Allocate train and test data into variables
    x_train = data['x_train']
    y_train = data['y_train']
    x_test  = data['x_test']
    y_test  = data['y_test']

    # Create and train the model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Get preditions and accuracy
    y_pred = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    # Save accuracy to file ==> Artifact
    with open(args.accuracy, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))

# Define main as a script start
if __name__ == '__main__':

    # Create arguments
    # IN: data
    # OUT: accuracy
    parser = argparse.ArgumentParser(description="Classify breast cancer type using Random Forest")
    parser.add_argument('--data', type=str)
    parser.add_argument('--accuracy', type=str)

    args = parser.parse_args()

    # Place data into specified in arguments file, directory is in 'create if not exists' mode
    Path(args.accuracy).parent.mkdir(parents=True, exist_ok=True)

    # Run the script
    _random_forest(args)