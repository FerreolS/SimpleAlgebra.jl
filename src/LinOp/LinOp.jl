
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end

function apply_adjoint(A::AbstractLinOp{I,O},x) where {I,O}
	@assert x ∈ O "The size of input parameter must be $(size(O))"
	apply_adjoint_(A,x) 
end

apply_adjoint_(A,x)  = apply_jacobian_(A,zeros(eltype(x),inputsize(A)),x)

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)

compose(A::AbstractLinOp,B::AbstractLinOp) = LinOpComposition(A, B)

function makeHtH(A::AbstractLinOp)
    throw(SimpleAlgebraFailure("unimplemented operation `makeHtH` for linear operator $(typeof(A))"))
end

include("./LinOpComposition.jl")
include("./LinOpDiag.jl")
include("./LinOpAdjoint.jl")
include("./LinOpDFT.jl")
include("./LinOpConv.jl")
