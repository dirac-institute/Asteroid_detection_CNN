"""Execute a notebook in-place with the kernel CWD pinned to the repo root.

The Evaluation notebooks use repo-root-relative paths (./models, ./DATA_DIFFIM, ...),
but they live in Evaluation/. nbconvert would run them with CWD=their own dir; nbclient's
resources metadata path lets us pin CWD to the repo root so the relative paths resolve.
"""
import sys, nbformat
from nbclient import NotebookClient

REPO = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
nb_path = sys.argv[1]
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                        resources={"metadata": {"path": REPO}})
client.execute()
nbformat.write(nb, nb_path)
print(f"RENDERED {nb_path}")
