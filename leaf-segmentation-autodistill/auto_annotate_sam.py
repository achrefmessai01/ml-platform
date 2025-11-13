from glob import glob

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

ontology = CaptionOntology(
    {
        "leaf": "leaves",
        "leaves": "leaves",
    }
)

base_model = GroundedSAM(ontology=ontology, box_threshold=0.3)

input_folder = "datasets/merged_data/segmentation/crop_cell_plus_disease/images"
output_folder = "datasets/merged_data/segmentation/crop_cell_plus_disease/images_labeled"

batches = glob(input_folder + "/*")

for batch in batches:
    print(f"Processing batch: {batch}")
    labels = base_model.label(input_folder=batch, extension=".jpg", output_folder=output_folder)
