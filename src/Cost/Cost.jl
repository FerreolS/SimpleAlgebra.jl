
abstract type AbstractCost{I}  <: AbstractMap{I,CoordinateSpace{(1,)}()}  end

include("./CostL2.jl")
