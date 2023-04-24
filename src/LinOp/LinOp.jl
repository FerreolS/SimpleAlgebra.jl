
abstract type AbstractLinOp{T,I,O}  <: AbstractMap{T,I,O}  end


function apply_adjoint(A::AbstractLinOp, ::AbstractVector) 
	throw(SimpleAlgebraFailure("unimplemented operation `apply` for mapping $(typeof(A))"))

end


function Base.adjoint(A::AbstractLinOp) 
	return LinOpAdjoint(A)
end


function ChainRulesCore.rrule( ::typeof(apply),A::AbstractLinOp, v)
    ∂Y(Δy) = (NoTangent(),NoTangent(), apply_adjoint(A,Δy))
    return apply(A,v), ∂Y
end


include("./LinOpDiag.jl")
include("./LinOpAdjoint.jl")