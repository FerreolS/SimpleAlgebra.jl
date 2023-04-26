
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end

function apply_jacobian(A::AbstractLinOp,v,x) 
	apply_adjoint(A,v)
end

function apply_adjoint(A::AbstractLinOp,) 
	throw(SimpleAlgebraFailure("unimplemented operation `apply_adjoint` for mapping $(typeof(A))"))

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