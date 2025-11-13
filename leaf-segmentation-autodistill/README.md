## Common issues

- grounding sam 2 uses florence to for object detection which is unfrtunately wrongly configured in autodistill package currently. To fix this you need to edit the file `venv/lib/python3.12/site-packages/autodistill_florence_2/model.py` and make the following changes:
    - Change the ontology prompt from 
        ```python
        "A photo of " + ", and ".join(ontology_classes) + "."
        ```
        to 
        ```python
        ", ".join(ontology_classes) + "."
        ```

- Grounding Sam 2 also wants to reinstall itself everytime the package is called. To fix this you need to edit the file `venv/lib/python3.12/site-packages/autodistill_grounded_sam_2/helpers.py` and make the following changes:
    - Change line 28 to 
        ```python
            if not os.path.exists(os.path.expanduser("~/.cache/autodistill/segment_anything_2/segment-anything-2")):
        ```