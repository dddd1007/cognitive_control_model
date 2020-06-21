# Set Path
import sys
sys.path.append('/Users/dddd1007/project2git/cognitive_control_model/models')
import cognitive_controller
# test the function of decision_maker

decision_maker = cognitive_controller.decision_maker()
decision_maker.receive_values([1,0,0,0], 1, [1,2,3,4], 5, 1)
decision_maker.decision()
decision_maker.decision(debug=False)

print('\033[1;32m=== Debug of decision_maker ===\033[0m')
print('The p_softmax is ' + format(decision_maker.p_softmax))

if decision_maker.p_softmax == 1:
    print("[!] The result of p_softmax is \033[1;32mright\033[0m")
else:
    print("[!] The result of p_softmax is \033[1;31mwrong\033[0m")

print("== The message blow must be '\033[1;31merror\033[0m' ==")
print(decision_maker.decision_maker_debug_dict)

decision_maker.decision(debug=True)
print("== This time will show right outputs: ==")
print(decision_maker.decision_maker_debug_dict)