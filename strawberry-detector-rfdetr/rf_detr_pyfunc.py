from rfdetr import RFDETRLarge
import mlflow.pyfunc
import numpy as np



class RFDetrPyFuncModel(mlflow.pyfunc.PythonModel):
    """A custom model that performs predictions using rfdetr.

    Args:
        PythonModel (mlflow.pyfunc.PythonModel): Base class for MLflow Python models.
    """

    def __init__(self):
        self.model = None

    def load_context(self, context):
        weights_path = context.artifacts["weights"]
        self.model = RFDETRLarge(pretrain_weights=weights_path)
        self.model.optimize_for_inference(compile=False)

    def predict(self, model_input, params):
        confidence_threshold = params.get("confidence_threshold")
        img = np.array(model_input)
        # Convert long types to int for detectron2 compatibility
        img = img.astype(np.uint8)

        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please ensure 'load_context' has been called."
            )
        detections = self.model.predict(img, threshold=confidence_threshold)
        return {
            "boxes": detections.xyxy,
            "scores": detections.confidence,
            "classes": detections.class_id,
        }
