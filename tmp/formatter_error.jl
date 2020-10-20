function update_options_weight_matrix(weight_vector::Array{Float64,1}, α::Float64,
                                      decay::Float64, correct_selection::Tuple;
                                      dodecay=true, debug=false)
    weight_matrix = reshape(weight_vector, 2, 2)'
    correct_selection_idx = CartesianIndex(correct_selection) + CartesianIndex(1, 1)
    op_selection_idx = abs(correct_selection[1] - 1) + 1

    if debug
        println("True selection is " * repr(correct_selection_idx))
        println("The value is " * repr(weight_matrix[correct_selection_idx]))
        println("True selection is " * repr(correct_selection_idx))
        println("The value is " * repr(weight_matrix[correct_selection_idx]))
    end

    weight_matrix[correct_selection_idx] = weight_matrix[correct_selection_idx] +
                                           α * (1 - weight_matrix[correct_selection_idx])

    if dodecay
        weight_matrix[op_selection_idx, :] = weight_matrix[op_selection_idx, :] .+
                                             decay .*
                                             (0.5 .- weight_matrix[op_selection_idx, :])
    end

    return reshape(weight_matrix', 1, 4)
end