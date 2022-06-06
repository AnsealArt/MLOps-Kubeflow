import kfp
from kfp import dsl

import argparse

from datetime import datetime

# Define components: preprocess, train, test, deploy

def preprocess_op(output_path, run_date, test_size):

    return dsl.ContainerOp(
        name = "Download and preprocess data",
        image = "dzieciolfilipit/kf_ch_preprocess_data:1.1",
        arguments = [
            '--output_path', output_path,
            '--run_date', run_date,
            '--test_size', test_size
        ],
        file_outputs = {
              'x_train': '{}{}/x_train.npy'.format(output_path, run_date)
            , 'x_test' : '{}{}/x_test.npy'.format(output_path, run_date)
            , 'y_train': '{}{}/y_train.npy'.format(output_path, run_date)
            , 'y_test' : '{}{}/y_test.npy'.format(output_path, run_date)
        }
    )

def train_op(x_train, y_train, output_path, run_date):

    return dsl.ContainerOp(
        name = "Train SGD Regressor",
        image = "dzieciolfilipit/kf_ch_train_model:1.1",
        arguments = [
            '--x_train', x_train,
            '--y_train', y_train,
            '--output_path', output_path,
            '--run_date', run_date
        ], 
        file_outputs = {
            'model_path': '{}{}/model.pkl'.format(output_path, run_date)
        }
    )

def test_op(x_test, y_test, model_path, output_path, run_date):

    return dsl.ContainerOp(
        name = "Test model and get MSE metric",
        image = "dzieciolfilipit/kf_ch_test_model:1.1",
        arguments = [
            '--x_test', x_test,
            '--y_test', y_test,
            '--model_path', model_path,
            '--output_path', output_path,
            '--run_date', run_date
        ],
        file_outputs = {
            'mean_squared_error': '{}{}/output.txt'.format(output_path, run_date)
        }
    )

def deploy_model_op(model_path, mse_path):

    return dsl.ContainerOp(
        name = "Deploy model for inference",
        image = "dzieciolfilipit/kf_ch_deploy_model:1.1",
        arguments = [
            '--model_path', model_path,
            '--mse_path', mse_path
        ]
    )


# Describe Pipeline
@dsl.pipeline(
    name = "California Housing prediction Pipeline",
    description = "Sample Pipeline for California Housing predictions using SGDRegressor regression model"
)
# Define Pipeline steps
def pipeline(test_size, output_path, deployment_threshhold_mse):

    # Use current date without separators
    # Sample date: 20220529160603
    run_date = datetime.now().strftime("%Y%m%d%H%M%S")

    _preprocess_op = preprocess_op(
        dsl.InputArgumentPath(output_path),
        dsl.InputArgumentPath(run_date),
        dsl.InputArgumentPath(test_size)
    )

    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train']),
        dsl.InputArgumentPath(output_path),
        dsl.InputArgumentPath(run_date)
    ).after(_preprocess_op)

    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model_path']),
        dsl.InputArgumentPath(output_path),
        dsl.InputArgumentPath(run_date)
    ).after(_test_op)

    # Define Production deployment condition, retrain otherwise
    with dsl.Condition(_test_op.outputs['mean_squared_error'] < deployment_threshhold_mse):

        deploy_model_op(
            dsl.InputArgumentPath(_train_op.outputs['model_path']),
            dsl.InputArgumentPath(_test_op.outputs['mean_squared_error'])
        ).after(_test_op)

    with dsl.Condition(_test_op.outputs['mean_squared_error'] >= deployment_threshhold_mse):

        client = kfp.Client(host='http://ml-pipeline-ui:80')
        client.create_run_from_pipeline_func(
            pipeline, 
            arguments = {
                'test_size': args.test_size,
                'output_path': args.output_path,
                'deployment_threshhold_mse': args.deployment_threshhold_mse
        }
    )



# Add arguments to main run
parser = argparse.ArgumentParser()

parser.add_argument('--test_size', type=float)
parser.add_argument('--output_path')
parser.add_argument('--deployment_threshhold_mse', type=float)

args = parser.parse_args()

# Create Kubeflow Client object and run Pipeline Function
client = kfp.Client(host='http://ml-pipeline-ui:80')
client.create_run_from_pipeline_func(
    pipeline, 
    arguments = {
        'test_size': args.test_size,
        'output_path': args.output_path,
        'deployment_threshhold_mse': args.deployment_threshhold_mse
    }
)