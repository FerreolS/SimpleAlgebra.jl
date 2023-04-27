struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{O,I,typeof(A)}(A)
end

function compose(A::LinOpAdjoint{I,O,D},B::D) where{I,O,D<:AbstractLinOp} 
	if A.parent===B
		return makeHtH(B)
	else
		throw(SimpleAlgebraFailure("unimplemented operation"))
	end
end

apply(A::LinOpAdjoint, v) = apply_adjoint(A.parent,v)

apply_adjoint(A::LinOpAdjoint, v) = apply(A.parent,v)
