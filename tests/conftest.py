import os

# Tests never write Red Alert match traces into the repo, and never call out to
# the live Jev API: tactician tests inject their own fake clients.
os.environ["RA_TRACE"] = "0"
os.environ["TRACE"] = "0"
os.environ["POKEMON_JEV"] = "0"
