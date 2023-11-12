
abstract type AbstractCost{I}  <: AbstractMap{I,Scalar{Real}}  end

outputspace(A::AbstractCost) = Scalar{Real}()

include("./CostL2.jl")
