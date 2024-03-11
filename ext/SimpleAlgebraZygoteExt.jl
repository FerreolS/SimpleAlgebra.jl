
module SimpleAlgebraZygoteExt
using SimpleAlgebra, Zygote, ChainRulesCore

function SimpleAlgebra.apply_jacobian(A::AbstractMap, v,x) 
	v ∈ I  || throw(SimpleAlgebraFailure("The size of the second parameter must be  $(size(I))"))
	x ∈ O  || throw(SimpleAlgebraFailure("The size of the third parameter must be $(size(O))"))
	if applicable(apply_jacobian_,A,v,x)
		return apply_jacobian_(A,v,x)
	else
		return Zygote.pullback(x->A*x,v)[2](x)[1]
	end
end

SimpleAlgebra.diff_via_ad(f,map::AbstractMap, v) = ChainRulesCore.rrule_via_ad(Zygote.ZygoteRuleConfig(),f,map, v)
end
