import mlflow.pyfunc
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detectron2PyFuncModel(mlflow.pyfunc.PythonModel):
    """A custom model that performs predictions using a Detectron2 based Faster R-CNN model.

    Args:
        PythonModel (mlflow.pyfunc.PythonModel): Base class for MLflow Python models.
    """

    def __init__(self):
        self.model = None

    def load_context(self, context):
        model_config = context.model_config or {"device": "cpu", "score_thresh": 0.5}
        weights_path = context.artifacts["weights"]
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.DEVICE = model_config["device"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_config.get("score_thresh", 0.5)
        cfg.MODEL.WEIGHTS = weights_path
        self.model = DefaultPredictor(cfg)

    def predict(self, model_input, params=None):
        img = np.array(model_input["image"])
        # Convert long types to int for detectron2 compatibility
        img = img.astype(np.uint8)

        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please ensure 'load_context' has been called."
            )
        outputs = self.model(img)
        instances = outputs["instances"].to("cpu")
        return {
            "boxes": instances.pred_boxes.tensor.numpy().tolist(),
            "scores": instances.scores.numpy().tolist(),
            "classes": instances.pred_classes.numpy().tolist(),
        }
