import os, sys, tempfile, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from voiceops.config import load_env, dataset_dir
if os.environ.get("VOICEOPS_NO_ENV_LOAD") != "1":
    load_env()
os.environ["TICKETS_PATH"] = os.path.join(tempfile.gettempdir(), "voiceops_pytest_tickets.csv")
try: dataset_dir(); HAVE_DATA = True
except FileNotFoundError: HAVE_DATA = False
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="set DATASET_DIR to the extracted dataset folder")

@pytest.fixture(scope="session")
def agent():
    from voiceops.agent import Agent
    return Agent(use_llm=False)
