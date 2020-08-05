# Set Path
import sys
sys.path.append('/Users/dddd1007/project2git/cognitive_control_model/models')

import pytest
import cognitive_controller

# test the function of decision_maker

def test_decision_maker():
    decision_maker = cognitive_controller.decision_maker()
    decision_maker.receive_values([1,0,0,0], 1, [1,2,3,4], 5, 1)
    decision_maker.decision()
    decision_maker.decision(debug=False)

    assert decision_maker.p_softmax == pytest.approx(1), decision_maker.decision_maker_debug_dict
