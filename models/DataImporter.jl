#=
Data importer for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

This model try to load research data for fitting
=#

module DataImporter

import DataFrames, CSV

function csv_load(filepath)
    CSV.File(filepath)
end

