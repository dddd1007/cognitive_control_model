using Models, StatsBase, Statistics, HypothesisTests
include("/Users/dddd1007/project2git/cognitive_control_model/tmp/init_sub1_data.jl")

dims = Dims((1,4))

result_stack = DataFrame(alpha = Float64[], alpha_CCC = Float64[], CCC = Float64[], decay = Float64[], 
                         corr_value = Float64[], corr_p = Float64[], AIC = Float64[], AIC_rl = Float64[])

#Threads.@threads for i = 1:1000
    a = collect(0.01:0.01:1)

    sample_rand = sample(a, dims)
    params_dict = Dict(zip([:α, :α_CCC, :CCC, :decay], sample_rand))

    results_model = model_recovery(sub1_env, sub1_subinfo, params_dict, model_type = :_1a1d1CCC)

    p_history = results_model[:p_selection_history]

    corr_result = CorrelationTest(p_history, sub1_subinfo.RT)

    corr_result.r
    corr_p = pvalue(corr_result)

    AIC = evaluate_relation(p_history, sub1_subinfo.RT)[:AIC]
    AIC_rl = 2*4 - 2*evaluate_relation(p_history, sub1_subinfo.RT)[:Loglikelihood]

    results_list = [sample_rand..., corr_result.r, corr_p, AIC, AIC_rl]

    push!(result_stack, results_list)
#end

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/relationship_corr_AIC.csv", result_stack)