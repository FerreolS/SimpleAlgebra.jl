

function apply_jacobian(A::AbstractMap{I,O},v,x )  where {I,O}
	@assert v ∈ I "The size of the second parameter must be  $(size(I))"
	@assert x ∈ O "The size of the third parameter must be $(size(O))"
	return Zygote.pullback(x->A*x,v)[2](x)[1]
#	throw(SimpleAlgebraFailure("unimplemented operation `apply_jacobian` for mapping $(typeof(A))"))
end
