struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{O,I,typeof(A)}(A)
end

function compose(A::LinOpAdjoint{I,O,D},B::D) where{I,O,D<:AbstractLinOp} 
	A.parent===B && return makeHtH(B)
	#throw(SimpleAlgebraFailure("unimplemented operation"))
	return LinOpComposition(A,B)
end

apply_(A::LinOpAdjoint, v) = apply_adjoint_(A.parent,v)

apply_adjoint_(A::LinOpAdjoint, v) = apply_(A.parent,v)

Base.adjoint(A::LinOpAdjoint) = A.parent	
