import argparse

# Model deployment actions to be provided later during the Project
def deploy_model(model_path, mse_path):
    print("==={} CAN BE DEPLOYED===".format(model_path))

    with open(mse_path, 'r') as file:
        print("Model SME: ".format(file.read()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path')
    parser.add_argument('--output_path')
    parser.add_argument('--run_date')
    args = parser.parse_args()

    deploy_model(args.model_path, args.output_path, args.run_date)