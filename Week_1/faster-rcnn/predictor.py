from detectron2.engine import DefaultPredictor

class CustomPredictor(DefaultPredictor):
    def __call__(self, original_image):
        outputs = super().__call__(original_image)
        instances = outputs["instances"]
        filtered_instances = self.filter_instances(instances)
        outputs["instances"] = filtered_instances
        return outputs

    def filter_instances(self, instances):
        """Keep only allowed classes: COCO class IDs (0=person, 2=car)"""
        pred_classes = instances.pred_classes.tolist()
        keep = [i for i, cls in enumerate(pred_classes) if cls in [0, 2]]
        filtered_instances = instances[keep]
        return filtered_instances
