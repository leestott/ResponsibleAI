from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import pandas as pd
from azureml.core import Model
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.core import Workspace
from azureml.explain.model._internal.explanation_client import ExplanationClient


SPLIT_SEED = 42
SPLIT_PERCENTAGE = 0.7
DATASET_PATH = '../../dataset/uci_dataset.csv'


def get_workspace():
    return Workspace.from_config("../notebooks-settings/config.json")


def test_model(fitted_model, test_ds):
    y = test_ds['target']
    X = test_ds.drop(['target'], axis=1)

    y_pred = fitted_model.predict(X)

    return {
        'AUC': roc_auc_score(y, y_pred),
        'Accuracy': accuracy_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
    }


if __name__ == "__main__":
    ws = get_workspace()
    df = pd.read_csv(DATASET_PATH)
    train_ds, test_ds = train_test_split(
        df, test_size=SPLIT_PERCENTAGE, random_state=SPLIT_SEED)

    automl_config = AutoMLConfig(name='Automated ML Experiment',
                                 task='classification',
                                 training_data=train_ds,
                                 validation_data=test_ds,
                                 label_column_name='target',
                                 iterations=1,
                                 primary_metric='AUC_weighted',
                                 max_concurrent_iterations=3,
                                 featurization='auto',
                                 model_explainability=True
                                 )

    automl_experiment = Experiment(ws, 'automl-classification')
    automl_run = automl_experiment.submit(automl_config)
    automl_run.wait_for_completion(show_output=True)

    best_run, fitted_model = automl_run.get_output()
    best_run_metrics = best_run.get_metrics()
    local_metrics = test_model(fitted_model, test_ds)

    client = ExplanationClient.from_run(best_run)
    engineered_explanations = client.download_model_explanation(raw=False)
    print(engineered_explanations.get_feature_importance_dict())


    print("AutoML metrics")
    for metric_name in best_run_metrics:
        print(f'{metric_name}: {best_run_metrics[metric_name]}')

    print("Local test metrics")
    for local_metric in local_metrics:
        print(f'{local_metric}: {local_metrics[local_metric]}')
