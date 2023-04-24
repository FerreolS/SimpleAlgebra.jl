
abstract type AbstractCost{T,I}  <: AbstractMap{T,I,(1,)}  end

include("./CostL2.jl")
