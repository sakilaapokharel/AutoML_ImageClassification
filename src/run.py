from automl import AutoML
from automl.search import Search
from automl.config import VISION_SEARCH
from automl.evaluator import VisionEvaluator


automl = AutoML(
    search=Search(VISION_SEARCH),
    evaluator=VisionEvaluator(),
    seed=42,
)


automl.fit(
    FlowersDataset
)


print(
    automl.get_best_config()
)