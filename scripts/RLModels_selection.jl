using Models

include("../tmp/init_sub1_data.jl")

model_evaluation(sub1_env, sub1_subinfo)

#fit_RL_SR(sub1_env, sub1_subinfo, 10, model_type=:basic)

result_list = zeros(4)
criteria = :AIC
optim_param_basic, _, _ = fit_RL_SR(env, realsub, 10, model_type=:basic)
p_history_basic = model_recovery(env, realsub, optim_param_basic)[:p_selection_history]
result_basic = evaluate_relation(p_history_basic, realsub.RT)[criteria]
result_table[1] = result_basic

optim_param_error, _, _ = fit_RL_SR(env, realsub, 100000, model_type=:error)
p_history_error = model_recovery(env, realsub, optim_param_error)[:p_selection_history]
result_error = evaluate_relation(p_history_error, realsub.RT)[criteria]
result_table[2] = result_error

optim_param_ccc_same, _, _ = fit_RL_SR(env, realsub, 1000000,
                                       model_type=:CCC_same_alpha)
p_history_ccc_same = model_recovery(env, realsub, optim_param_ccc_same)[:p_selection_history]
result_ccc_same = evaluate_relation(p_history_ccc_same, realsub.RT)[criteria]
result_table[3] = result_ccc_same

optim_param_ccc_diff, _, _ = fit_RL_SR(env, realsub, 5000000,
                                       model_type=:CCC_different_alpha)
p_history_ccc_diff = model_recovery(env, realsub, optim_param_ccc_diff)[:p_selection_history]
result_ccc_diff = evaluate_relation(p_history_ccc_diff, realsub.RT)[criteria]
result_table[4] = result_ccc_diff