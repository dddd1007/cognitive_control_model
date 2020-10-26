### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ fd156d94-14e9-11eb-367c-4d69767436df
using DataFrames, CSV

# ╔═╡ bb3be282-1506-11eb-39d9-b5eeeb38c296
using Statistics

# ╔═╡ 9394af46-1503-11eb-3793-9bf76150e79e
# 导入全部数据
include("../scripts/import_all_data.jl")

# ╔═╡ 3bb97d24-14e5-11eb-17b2-17e2e46e812c
md"## 整合数据"

# ╔═╡ 8ad08f4c-14e5-11eb-3895-271c7ad84a11
# import Data
csv_files = "../data/output/RLModels/model_selection/"

# ╔═╡ af564ae2-14e9-11eb-25e1-01ffdad6597c
file_list = readdir(csv_files);

# ╔═╡ d737367a-14e9-11eb-2f98-670b22e03c80
model_selection_result = DataFrame(zeros(5)', [:Sub, :basic, :error, :same_CCC, :diff_CCC])

# ╔═╡ 2793457a-14ed-11eb-3757-b799ddb8c341
model_selection_result.Sub = string(model_selection_result.Sub)

# ╔═╡ b3f59134-14e9-11eb-0f58-17c30084d6ed
function combinetable(x, csv_files, file_list)
	for i in 1:length(file_list)
		single_file = csv_files * file_list[i]
		temp_file = DataFrame(CSV.read(single_file))
		temp_file.Sub = replace(file_list[i], ".csv"=>"")
		x = vcat(x, temp_file)
	end
	delete!(x,1)
	return x
end

# ╔═╡ cf44635c-14ee-11eb-07ca-25ccdc33b060
result = combinetable(model_selection_result, csv_files, file_list)

# ╔═╡ 5e6f11f0-14f2-11eb-1cdc-19b8fa33fca8
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/RLModel_selection.csv", result)

# ╔═╡ 03b8b5ac-14f4-11eb-1726-85e690c495eb
md"## 查看错误率"

# ╔═╡ ce36e52e-1503-11eb-0170-4d8f272a3e61
grouped_data = groupby(all_data, :Subject)

# ╔═╡ 3f65b918-1505-11eb-2249-71025bcb5396
names(grouped_data)

# ╔═╡ d3f8207c-1503-11eb-1dd9-2107385bbe34
error_rate = combine(grouped_data, :Type => mean)

# ╔═╡ eda58d92-1507-11eb-2929-d7e44a5f07ce
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/error_rate.csv", error_rate)

# ╔═╡ Cell order:
# ╠═3bb97d24-14e5-11eb-17b2-17e2e46e812c
# ╠═fd156d94-14e9-11eb-367c-4d69767436df
# ╠═8ad08f4c-14e5-11eb-3895-271c7ad84a11
# ╠═af564ae2-14e9-11eb-25e1-01ffdad6597c
# ╠═d737367a-14e9-11eb-2f98-670b22e03c80
# ╠═2793457a-14ed-11eb-3757-b799ddb8c341
# ╠═b3f59134-14e9-11eb-0f58-17c30084d6ed
# ╠═cf44635c-14ee-11eb-07ca-25ccdc33b060
# ╠═5e6f11f0-14f2-11eb-1cdc-19b8fa33fca8
# ╠═03b8b5ac-14f4-11eb-1726-85e690c495eb
# ╠═9394af46-1503-11eb-3793-9bf76150e79e
# ╠═ce36e52e-1503-11eb-0170-4d8f272a3e61
# ╠═3f65b918-1505-11eb-2249-71025bcb5396
# ╠═bb3be282-1506-11eb-39d9-b5eeeb38c296
# ╠═d3f8207c-1503-11eb-1dd9-2107385bbe34
# ╠═eda58d92-1507-11eb-2929-d7e44a5f07ce
