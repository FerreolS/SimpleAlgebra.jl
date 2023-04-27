
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end

apply_adjoint(A::AbstractLinOp,x) = apply_jacobian(A,zeros(eltype(x),inputsize(A)),x)

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)

include("./LinOpDiag.jl")
include("./LinOpAdjoint.jl")
