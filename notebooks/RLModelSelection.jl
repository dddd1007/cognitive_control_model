### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ fd156d94-14e9-11eb-367c-4d69767436df
using DataFrames, CSV

# ╔═╡ 3bb97d24-14e5-11eb-17b2-17e2e46e812c
md"整合数据“

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

# ╔═╡ afea5746-14f3-11eb-00d0-29662b53ea11
vcat(model_selection_result, temp_file)

# ╔═╡ 03b8b5ac-14f4-11eb-1726-85e690c495eb
model_selection_result

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
# ╠═afea5746-14f3-11eb-00d0-29662b53ea11
# ╠═03b8b5ac-14f4-11eb-1726-85e690c495eb
