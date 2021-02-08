using BenchmarkTools
using PyCall
import Pkg

ENV["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
Pkg.build("RCall")
using RCall

a = rand(10^7)

@btime pybuiltin("sum")(a)
@btime R"sum($a)"
@btime sum(a)