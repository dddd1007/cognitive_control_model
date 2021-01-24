using Models, DataFrames, Hyperopt

include("/Users/dddd1007/project2git/cognitive_control_model/tmp/init_sub1_data.jl")

v_list = []
s_list = []
mse_list = []

ho = @hyperopt for i = 10000,
        α_v = [0.001:0.001:1;],
        α_s = [0.001:0.001:1;]

    agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
    model_stim = RLModels.NoSoftMax.rl_learning_sr(sub1_env, agent, sub1_subinfo, dodecay = false)
    mse = evaluate_relation(model_stim[:p_selection_history], sub1_subinfo.RT)[:MSE]
    push!(v_list, α_v)
    push!(s_list, α_s)
    push!(mse_list, mse)
end

gradients_table = DataFrame(alpha_v = v_list, alpha_s = s_list, mse = mse_list)

using Gadfly
p = plot(gradients_table, x = :alpha_v, y = :alpha_s, color = :mse, Geom.point)
q = plot(gradients_table, x = :mse, Geom.histogram)