from model import AutoML
from dotenv import load_dotenv
import os

load_dotenv()

print("API key loaded:", os.getenv("TABPFN_API_KEY") is not None)

os.environ["TABPFN_TOKEN"] = os.getenv("TABPFN_API_KEY")


automl = AutoML(
    dataset="flowers",
    fidelity=29,
    tabpfn_mode="client",
)


automl.search_config()
