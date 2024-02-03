
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end

function apply_adjoint(A::AbstractLinOp{I,O},x) where {I,O}
	x ∈ outputspace(A) || throw(SimpleAlgebraFailure("The size of input parameter must be $(outputsize(A))")) 
	if applicable(apply_adjoint_,A,x) 
		return apply_adjoint_(A,x) 
	else
		return apply_jacobian(A,zeros(eltype(x),inputsize(A)),x)
	end
end

function apply_adjoint!(S::Scratchspace,A::AbstractMap{I,O}, x) where {I,O}
	x ∈ outputspace(A) || throw(SimpleAlgebraFailure("The input size must belong to the space $(inputspace(A))")) 
	y = newarray!(S,inputspace(A))
	apply_adjoint_!(y,A, x)
end

function apply_adjoint_ end

apply_jacobian(A::AbstractLinOp{I,O}, _,x) where {I,O} = apply_adjoint(A,x) 
# apply_adjoint(A,x::AbstractLinOp)  = apply_jacobian(A,zeros(eltype(x),inputsize(A)),x)



# FIXME issue here should be generated only when apply_adjoint is implemented
#function ChainRulesCore.rrule(config::RuleConfig{>:HasForwardsMode}, ::typeof(apply),A::AbstractLinOp, v)
function ChainRulesCore.rrule( ::typeof(apply),A::AbstractLinOp, v)
	if applicable(apply_adjoint_,A,v)
    	∂Y(Δy) = (NoTangent(),NoTangent(), apply_adjoint_(A,Δy))
    	return apply(A,v), ∂Y
	else
		return ChainRulesCore.rrule_via_ad(config,apply_,A, v)
	end
end

function ChainRulesCore.rrule( ::typeof(apply_adjoint),A::AbstractLinOp, v)
	if applicable(apply_,A,v)
		∂Y(Δy) = (NoTangent(),NoTangent(), apply_(A,Δy))
		return apply_adjoint(A,v), ∂Y
	else
		return ChainRulesCore.rrule_via_ad(config,apply_adjoint_,A, v)
	end
end



include("./OperationsOnLinOp.jl")
include("./LinOpDiag.jl")
include("./LinOpDFT.jl")
include("./LinOpConv.jl")
include("./LinOpSelect.jl")
include("./LinOpGrad.jl")
