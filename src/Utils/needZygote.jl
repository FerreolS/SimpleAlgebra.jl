

function apply_jacobian(A::AbstractMap, v,x) 
	#@assert v ∈ I "The size of the second parameter must be  $(size(I))"
	#@assert x ∈ O "The size of the third parameter must be $(size(O))"
	if applicable(apply_jacobian_,A,v,x)
		return apply_jacobian_(A,v,x)
	else
		return Zygote.pullback(x->A*x,v)[2](x)[1]
	end
end
