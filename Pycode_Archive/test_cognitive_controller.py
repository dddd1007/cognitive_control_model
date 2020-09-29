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
    data_dict = {'sub': 'Subject', 'loc': 'location', 'RT': 'RT'}
    data_table = pd.read_csv(csv_file_path)
    data_reader = cognitive_controller.data_reader(csv_file = csv_file_path, data_dict = data_dict)
    data_expect = {'RT': np.array(data_table['RT']), 'loc': np.array(data_table['location'])}

    # 测试数据读取的正确性
    pd.testing.assert_frame_equal(data_reader.data, data_table)

    # 测试数据提取的正确性
    extracted_data = data_reader.extract_data('RT', 'loc')
    assert extracted_data.keys() == data_expect.keys()
    assert np.all(extracted_data['RT'] == data_expect['RT'])


def test_bayesian_learner():
    csv_file_path = "/Users/dddd1007/project2git/cognitive_control_model/data/unit_test/test_table.csv"
    data_dict = {'sub': 'Subject', 'loc': 'location', 'RT': 'RT', 'contingency': 'contingency'}
    data_reader = cognitive_controller.data_reader(csv_file = csv_file_path, data_dict = data_dict)
    input_data = data_reader.extract_data("RT", "loc")
    bayesian_learner = cognitive_controller.Bayesian_learner(input_data)


def test_bayesian_sc_learner():
    pass

def test_value_convertor():
    input_array = ['a','b','b','a']
    matchup_dict = {"a": 1, 'b': 2}
    result = cognitive_controller.value_convertor(input_array, matchup_dict)

    message = "The type of result is " + format(type(result))
    assert np.all(result == np.array([1,2,2,1])), message