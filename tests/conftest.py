import os

# Tests never write Red Alert match traces into the repo.
os.environ["RA_TRACE"] = "0"
