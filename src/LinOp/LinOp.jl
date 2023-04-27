
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end


function apply_adjoint(A::AbstractLinOp,x) 
	apply_jacobian(A,zeros(eltype(x),sizein(A)),x)
end



function Base.adjoint(A::AbstractLinOp) 
	return LinOpAdjoint(A)
end




include("./LinOpDiag.jl")
include("./LinOpAdjoint.jl")
