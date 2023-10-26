

function apply_jacobian_(A::AbstractMap,v,x )  
	return Zygote.pullback(x->A*x,v)[2](x)[1]
end
