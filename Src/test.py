import pytest
from preprocessing import universe_select


def test_import_data():
    """Checks that the data is loaded into a
    dictionary and that it is not empty"""
    path = "Data/Commodity_Data/"
    universe_dict = universe_select(path, "Cu")
    assert(type(universe_dict) == dict)
    assert (bool(universe_dict) == True)
    