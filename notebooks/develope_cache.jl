### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 383dd75c-060e-11eb-1fd1-85f6eb33a9af
using DataFrames, DataFramesMeta, CSV, Statistics, StatsBase

# ╔═╡ 49632e92-060e-11eb-05a8-9b0fccf669bd
include("/Users/dddd1007/project2git/cognitive_control_model/models/DataImporter.jl")

# ╔═╡ a60d9102-060a-11eb-1c04-fdb0ce5006ac
md"# 编写模型各功能"

# ╔═╡ ec09183e-060a-11eb-2690-c9aa2c7e2a31
md"## 测试DataImporter模块； 导入数据"

# ╔═╡ 30a9fb66-0644-11eb-08d7-310a38336154
#导入一个被试的数据开始分析
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv");

# ╔═╡ 64a66ae6-077d-11eb-2e8c-5bdbc301926d
head(all_data)

# ╔═╡ 91240684-060e-11eb-3e4a-d1c1878d8c3a
begin
	color_rule = Dict("red" => "0" , "green" => "1")
	congruency_rule = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
	loc_rule = Dict("left" => "0", "right" => "1")
	transform_rule = Dict("stim_color" => color_rule, "Type" => Type_rule, 
		"stim_loc" => loc_rule, "congruency" => congruency_rule)
end

# ╔═╡ ba1d75b4-077d-11eb-22de-e35a8ca85485
transform_data!(all_data, transform_rule)

# ╔═╡ 5cb10ab4-07d8-11eb-323a-355d87f5075d
sub1_data = @where(all_data, :Subject .== "sub01_Yangmiao");

# ╔═╡ 8360a24a-062d-11eb-2ae2-891d1573864c
begin
    env_idx_dict = Dict("stim_task_related" => "stim_color", 
		                "stim_task_unrelated" => "stim_loc", 
		                "stim_action_congruency" => "congruency", 
		                "correct_action" => "correct_action",
		                "env_type" => "condition", "sub_tag" => "Subject")
	sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
		                "corrections" => "Type", "sub_tag" => "Subject")
end

# ╔═╡ a4f8b8c6-0715-11eb-3453-9d4442e54113
sub1_env, sub1_subinfo = init_env_sub(sub1_data, env_idx_dict, sub_idx_dict)

# ╔═╡ 622e58a4-087f-11eb-12aa-c9e85acc2d8e
md"## 强化学习模型"

# ╔═╡ 9a0ce0ec-09db-11eb-3142-2d5c455cf2f8
md"### 模型1 学习 S-R 联结的强化学习模型, 使用 SoftMax 决策, 带 Decay"

# ╔═╡ 7434237e-087f-11eb-02fd-75fd84662051
# 学习抽象概念的基本对象
struct SRLearner_basic
	α_v::Float64
	β_v::Float64
	α_s::Float64
	β_s::Float64
	decay::Float64
end

# ╔═╡ 776783b6-089c-11eb-2e3c-fb91cb8808fe
# 定义SR学习中的SoftMax
function sr_softmax(options_vector::Array{Float64,1}, β::Float64, 
		true_selection::Tuple, debug=false)
	options_matrix = reshape(options_vector, 2, 2)'

	op_selection_idx = CartesianIndex(true_selection[1], abs(true_selection[2] - 1)) + CartesianIndex(1,1)
	true_selection_idx = CartesianIndex(true_selection) + CartesianIndex(1,1)
	
	if debug
		println(options_matrix)
	 	println("True selection is " * repr(options_matrix[true_selection_idx]))
	 	println("Op selection is " * repr(options_matrix[op_selection_idx]))
	end
	
	exp(β * options_matrix[true_selection_idx])/
	(exp(β * options_matrix[true_selection_idx]) + exp(β * options_matrix[op_selection_idx]))
end

# ╔═╡ 85686c42-08a5-11eb-24b7-3598b5697559
# 定义更新 Weight 的矩阵
function update_options_weight_matrix(weight_vector::Array{Float64,1}, α::Float64, 
		decay::Float64, correct_selection::Tuple; dodecay=true, debug=false)
	weight_matrix = reshape(weight_vector, 2, 2)'
	correct_selection_idx = CartesianIndex(correct_selection) + CartesianIndex(1,1)
	op_selection_idx = abs(correct_selection[1] - 1) + 1
	
	if debug
		println("True selection is " * repr(correct_selection_idx))
		println("The value is " * repr(weight_matrix[correct_selection_idx]))
	end
	
	weight_matrix[correct_selection_idx] = weight_matrix[correct_selection_idx] + α * (1-weight_matrix[correct_selection_idx])
	
	if dodecay
		weight_matrix[op_selection_idx,:] = weight_matrix[op_selection_idx,:] .+ 
		decay .* (0.5 .- weight_matrix[op_selection_idx,:])
	end
	
	return weight_matrix
end

# ╔═╡ e22061a2-09f8-11eb-2db0-bfbe7004d73d
# 定义评估变量相关性的函数
function evaluate_relation(x, y, method)
	if method == "mse"
		return sum(abs2.(x .- y))
	elseif method == "cor"
		return cor(x, y)
	end
end

# ╔═╡ cf78fe4e-0ac8-11eb-2192-216afb60ddb9
# 初始化更新矩阵和基本参数
function init_param(env, agent, learn_type="sr")
	
    total_trials_num = length(env.stim_task_unrelated)
	
	if learn_type == "sr"
		options_weight_matrix = zeros(Float64, (total_trials_num + 1, 4))
		options_weight_matrix[1,:] = [0.5,0.5,0.5,0.5]
	elseif learn_type == "abstract_concept"
		options_weight_matrix = zeros(Float64, (total_trials_num + 1, 2))
	    options_weight_matrix[1,:] = [0.5,0.5]
	end

	p_softmax_history = zeros(Float64, total_trials_num)
	α = 0.0
	β = 0.0
	decay = agent.decay

	return (total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay)
end

# ╔═╡ 47e4d094-0892-11eb-320b-739a0a7cc364
# 定义强化学习函数
function rl_learning(env::ExpEnv, agent::SRLearner_basic, realsub::RealSub; 
		eval_method = "mse" ,verbose = false)
	
	# Check the subtag
	if env.sub_tag != realsub.sub_tag
		return println("The env and sub_real_data not come from the same one!")
	end
	
	# init learning parameters list
	total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay = init_param(env, agent)
	
	# Start learning
	for idx in 1:total_trials_num
		
		if env.env_type[idx] == "v"
			β = agent.β_v
			α = agent.α_v
		elseif env.env_type[idx] == "s"
			β = agent.β_s
			α = agent.α_s
		end
		
		## Decision
		p_softmax_history[idx] = sr_softmax(options_weight_matrix[idx,:], β, 
				(env.stim_task_unrelated[idx], realsub.response[idx]))
			
		## Update 
		options_weight_matrix[idx+1,:] = 
			update_options_weight_matrix(options_weight_matrix[idx,:], α, decay, 
				(env.stim_task_unrelated[idx], env.stim_correct_action[idx]))'
	end
	
	# Evaluate result
	eval_result = evaluate_relation(realsub.RT, p_softmax_history, eval_method)
	
	if !verbose
		result = [agent.α_v, agent.β_v, agent.α_s, 
			agent.β_s, agent.decay, eval_result]
		return result
	elseif verbose
		return ("options_weight_matrix" => options_weight_matrix, 
			"p_softmax_history" => p_softmax_history)
	end
end

# ╔═╡ fb79a89c-09db-11eb-15e8-6d7305e1a0f2
md"### 模型2 学习 S-R 联结的强化学习模型, 使用 SoftMax 决策, 错误试次下学习率不同"

# ╔═╡ 6adf9fb6-09f0-11eb-26bb-77b2afd45c24
# 学习抽象概念的基本对象
struct SRLearner_witherror
	basic_learner::SRLearner_basic
	error_learner::SRLearner_basic
end

# ╔═╡ bc5253a8-0ad2-11eb-069e-a332aff4bb20


# ╔═╡ 2a26c3fa-09f5-11eb-3850-6dacc388c686
# 定义强化学习函数
function rl_learning(env::ExpEnv, agent::SRLearner_witherror, realsub::RealSub; 
		verbose = false)
	
	# Check the subtag
	if env.sub_tag != realsub.sub_tag
		return println("The env and sub_real_data not come from the same one!")
	end
	
	# init learning parameters list
	total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay = init_param(env, agent)
	
	# Start learning
	for idx in 1:total_trials_num
		
		if env.env_type[idx] == "v" 
			if realsub.corrections[idx] == 1
				β = agent.basic_learner.β_v
				α = agent.basic_learner.α_v
			elseif realsub.corrections[idx] == 0
				β = agent.error_learner.β_v
				α = agent.error_learner.α_v				
			end
		elseif env.env_type[idx] == "s"
			if realsub.corrections[idx] == 1
				β = agent.basic_learner.β_s
				α = agent.basic_learner.α_s
			elseif realsub.corrections[idx] == 0
				β = agent.error_learner.β_s
				α = agent.error_learner.α_s				
			end
		end
		
		## Decision
		p_softmax_history[idx] = sr_softmax(options_weight_matrix[idx,:], β, 
				(env.stim_task_unrelated[idx], realsub.response[idx]))
			
		## Update 
		options_weight_matrix[idx+1,:] = 
			update_options_weight_matrix(options_weight_matrix[idx,:], α, decay, 
				(env.stim_task_unrelated[idx], env.stim_correct_action[idx]))'
	end
	
	# Evaluate result
	eval_result = evaluate_relation(realsub.RT, p_softmax_history, eval_method)
	
	if !verbose
		result = [agent.α_v, agent.β_v, agent.α_s, agent.β_s, mse]
		return result
	elseif verbose
		return options_weight_matrix[total_trials_num, :]
	end
end

# ╔═╡ 46a7cac6-0acf-11eb-2ba8-a175c4a140e9
begin
	α_v = rand([0.1:0.01:1.0;])
	β_v = rand([0.1:0.01:5.0;])
	α_s = rand([0.1:0.01:1.0;])
	β_s = rand([0.1:0.01:5.0;])
	decay = rand([0.1:0.01:1.0;])
	sub1_agent = SRLearner_basic(α_v, β_v, α_s, β_s, decay)
	rl_learning(sub1_env, sub1_agent, sub1_subinfo)
end

# ╔═╡ e53f76ee-08ad-11eb-00fd-8b7ea211b136
begin
	number_iterations = 10000
	result_table = zeros(Float64, (number_iterations,6))
	Threads.@threads for i in 1:number_iterations
		α_v = rand([0.1:0.01:1.0;])
		β_v = rand([0.1:0.01:5.0;])
		α_s = rand([0.1:0.01:1.0;])
		β_s = rand([0.1:0.01:5.0;])
		decay = rand([0.1:0.01:1.0;])
		sub1_agent = SRLearner_basic(α_v, β_v, α_s, β_s, decay)
		result_table[i,:] = rl_learning(sub1_env, sub1_agent, sub1_subinfo)
	end
end

# ╔═╡ 2412cc44-0ac3-11eb-337e-cd2299f61f01
StatsBase.summarystats(result_table[:,6])

# ╔═╡ Cell order:
# ╠═a60d9102-060a-11eb-1c04-fdb0ce5006ac
# ╟─ec09183e-060a-11eb-2690-c9aa2c7e2a31
# ╠═383dd75c-060e-11eb-1fd1-85f6eb33a9af
# ╠═49632e92-060e-11eb-05a8-9b0fccf669bd
# ╠═30a9fb66-0644-11eb-08d7-310a38336154
# ╠═64a66ae6-077d-11eb-2e8c-5bdbc301926d
# ╠═91240684-060e-11eb-3e4a-d1c1878d8c3a
# ╠═ba1d75b4-077d-11eb-22de-e35a8ca85485
# ╠═5cb10ab4-07d8-11eb-323a-355d87f5075d
# ╠═8360a24a-062d-11eb-2ae2-891d1573864c
# ╠═a4f8b8c6-0715-11eb-3453-9d4442e54113
# ╠═622e58a4-087f-11eb-12aa-c9e85acc2d8e
# ╠═9a0ce0ec-09db-11eb-3142-2d5c455cf2f8
# ╠═7434237e-087f-11eb-02fd-75fd84662051
# ╠═776783b6-089c-11eb-2e3c-fb91cb8808fe
# ╠═85686c42-08a5-11eb-24b7-3598b5697559
# ╠═e22061a2-09f8-11eb-2db0-bfbe7004d73d
# ╠═cf78fe4e-0ac8-11eb-2192-216afb60ddb9
# ╠═47e4d094-0892-11eb-320b-739a0a7cc364
# ╠═46a7cac6-0acf-11eb-2ba8-a175c4a140e9
# ╠═e53f76ee-08ad-11eb-00fd-8b7ea211b136
# ╠═2412cc44-0ac3-11eb-337e-cd2299f61f01
# ╠═fb79a89c-09db-11eb-15e8-6d7305e1a0f2
# ╠═6adf9fb6-09f0-11eb-26bb-77b2afd45c24
# ╠═bc5253a8-0ad2-11eb-069e-a332aff4bb20
# ╠═2a26c3fa-09f5-11eb-3850-6dacc388c686
