import pytest
import numpy as np
import pandas as pd
from models import cognitive_controller


def test_decision_maker():
    decision_maker = cognitive_controller.decision_maker()
    decision_maker.receive_values([1, 0, 0, 0], 1, [1, 2, 3, 4], 5, 1)
    decision_maker.decision()
    decision_maker.decision(debug = False)

    assert decision_maker.p_softmax == pytest.approx(1), decision_maker.decision_maker_debug_dict


def test_data_reader():
    csv_file_path = "/Users/dddd1007/project2git/cognitive_control_model/data/unit_test/test_table.csv"
    data_dict = {'Sub': 'Subject', 'loc': 'location', 'RT': 'RT'}
    data_reader = cognitive_controller.data_reader(csv_file = csv_file_path, data_dict = data_dict)

    pd.testing.assert_frame_equal(data_reader.data, pd.read_csv(csv_file_path))

data_reader.extract_data("RT", "loc")
