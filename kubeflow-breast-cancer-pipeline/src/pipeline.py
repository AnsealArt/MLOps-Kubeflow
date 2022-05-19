import kfp
from kfp import dsl
from kfp.components import func_to_container_op

@func_to_container_op
def show_results(decision_tree : float, logistic_regression : float, random_forest : float) -> None:
    
    # Show metrics summary for all the models
    print('=== Accuracy for chosen models ===')
    print(f'Decision Tree: {decision_tree}')
    print(f'Logistic Regression: {logistic_regression}')
    print(f'Random Forest: {random_forest}')

@dsl.pipeline(name='Breast Cancer Classification Pipeline', description='Applies ML models to classify breast cancer type')
def breast_cancer_classification_pipeline():

    # Load YAML files for each component
    download_data = kfp.components.load_component_from_file('download_data/download_data.yaml')
    decision_tree = kfp.components.load_component_from_file('decision_tree/decision_tree.yaml')
    logistic_regression = kfp.components.load_component_from_file('logistic_regression/logistic_regression.yaml')
    random_forest = kfp.components.load_component_from_file('random_forest/random_forest.yaml')

    # Set pipeline tasks and pass data to models
    download_task = download_data()
    decision_tree_task = decision_tree(download_task.output)
    logistic_regression_task = logistic_regression(download_task.output)
    random_forest_task = random_forest(download_task.output)

    # Show results of ML models
    show_results(
        decision_tree_task.output,
        logistic_regression_task.output,
        random_forest_task.output
    )

# Compile the pipeline so it can be imported into Kubeflow
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(breast_cancer_classification_pipeline, 'BreastCancerClassificationPipeline.yaml')