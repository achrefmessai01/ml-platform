import mlflow
import json
import os
import cv2
from datetime import datetime
from rfdetr import RFDETRLarge
from rf_detr_pyfunc import RFDetrPyFuncModel
from mlflow.models import infer_signature

os.environ["MLFLOW_TRACKING_TOKEN"] = ""

mlflow_config = {
    "experiment_name": "rfdetr_pepper_detector_single_class",
    "tracking_uri": "https://gitlab.com/api/v4/projects/72607084/ml/mlflow/",
}

# Hyperparameters
hyperparameters = {
    "base_model_name": "RFDETRMedium",
    "registered_model_name": "rfdetr_pepper_detector_single_class",
    "dataset_path": "pepper",
    "output_dir": "output/rfdetr_pepper_detector_single_class",
    "epochs": 40,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "device": "cuda",
    "early_stopping": True,
    "early_stopping_patience": 5,
    "early_stopping_use_ema": True,
    "learning_rate": 1e-4,
    "resolution": 800,
}


def train():
    model = RFDETRLarge()
    model.train(
        dataset_dir=hyperparameters["dataset_path"],
        epochs=hyperparameters["epochs"],
        batch_size=hyperparameters["batch_size"],
        grad_accum_steps=hyperparameters["grad_accum_steps"],
        output_dir=hyperparameters["output_dir"],
        early_stopping=hyperparameters["early_stopping"],
        early_stopping_patience=hyperparameters["early_stopping_patience"],
        lr=hyperparameters["learning_rate"],
        device=hyperparameters["device"],
        early_stopping_use_ema=hyperparameters["early_stopping_use_ema"],
        resolution=hyperparameters["resolution"],
    )


def log_experiment():
    """Log experiment details to MLflow."""
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run():
        # Log hyperparameters
        for param, value in hyperparameters.items():
            mlflow.log_param(param, value)

        # Log metrics from output folder
        metric_file = f"{hyperparameters['output_dir']}/log.txt"
        with open(metric_file, "r") as f:
            for line in f:
                dict_line = json.loads(line)

                # Extract and convert timestamp
                timestamp = dict_line.pop("now_time", None)
                timestamp = (
                    int(round(datetime.fromisoformat(timestamp).timestamp()))
                    if timestamp
                    else None
                )

                epoch = dict_line.pop("epoch", None)
                test_results_dict = dict_line.pop("test_results_json", None)

                # Log time metrics
                for time_key in ["train_epoch_time", "epoch_time"]:
                    if time_value := dict_line.pop(time_key, None):
                        mlflow.log_metric(
                            time_key,
                            int(datetime.strptime(time_value, "%H:%M:%S").timestamp()),
                            step=epoch,
                            timestamp=timestamp,
                        )

                # Log class-specific metrics
                if class_map := test_results_dict.pop("class_map", None):
                    for item in class_map:
                        class_name = item.pop("class")
                        for key, value in item.items():
                            mlflow.log_metric(
                                f"{class_name}_{key}".replace("@", "_"),
                                value,
                                step=epoch,
                                timestamp=timestamp,
                            )

                # Remove unwanted keys
                for key in [
                    "ema_test_coco_eval_bbox",
                    "test_coco_eval_bbox",
                    "ema_test_results_json",
                ]:
                    dict_line.pop(key, None)

                # Log test results
                for key, value in test_results_dict.items():
                    mlflow.log_metric(key, value, step=epoch, timestamp=timestamp)

                # Log remaining metrics
                for key, value in dict_line.items():
                    mlflow.log_metric(key, value, step=epoch, timestamp=timestamp)

        # Log artifacts
        mlflow.log_artifact(f"{hyperparameters['output_dir']}/results.json")
        mlflow.log_artifact(f"{hyperparameters['output_dir']}/metrics_plot.png")
        mlflow.log_artifact(metric_file)

        # Log model
        model_path = f"{hyperparameters['output_dir']}/checkpoint_best_total.pth"
        mlflow.log_artifact(model_path, "model_weights")
        
        # Infer signature
        model = RFDETRLarge(pretrain_weights=model_path)
        image = cv2.imread("test.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = model.predict(image, threshold=0.5)

        output_example = {
            "boxes": detections.xyxy,
            "scores": detections.confidence,
            "classes": detections.class_id,
        }

        params = {"confidence_threshold": 0.5}

        signature = infer_signature(image, output_example, params)
        
        conda_env = {
            "channels": ["pytorch", "conda-forge"],
            "dependencies": [
                "python=3.12.3",
                "pip",
                "pytorch",
                "torchvision",
                {
                    "pip": [
                        "mlflow",
                        "rfdetr",
                    ]
                },
            ],
            "name": "mlflow-env",
        }

        mlflow.pyfunc.log_model(
            python_model=RFDetrPyFuncModel(),
            name="rfdetr_pepper_detector_single_class_model",
            registered_model_name=hyperparameters["registered_model_name"],
            artifacts={"weights": model_path},
            conda_env=conda_env,
            code_paths=["rf_detr_pyfunc.py"],
            signature=signature,
        )

        # Clean up output directory
        os.remove("output")


if __name__ == "__main__":
    train()
    log_experiment()
