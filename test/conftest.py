import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runplot", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "plot: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runplot"):
        # --runplot given in cli: do not skip slow tests
        return
    skip_plot = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "plot" in item.keywords:
            item.add_marker(skip_plot)
