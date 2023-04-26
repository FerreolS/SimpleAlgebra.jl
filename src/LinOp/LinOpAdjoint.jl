struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{O,I,typeof(A)}(A)
end

LinOpAdjoint(A::LinOpAdjoint)  =  A

function compose(A::LinOpAdjoint{I,O,D},B::D) where{I,O,D<:AbstractLinOp} 
	if A.parent===B
		return makeHtH(B)
	else
		throw(SimpleAlgebraFailure("unimplemented operation"))
	end
end

function apply(A::LinOpAdjoint, v) 
	return apply_adjoint(A.parent,v)
end

function apply_adjoint(A::LinOpAdjoint, v) 
	return apply(A.parent,v)
end

# #@opt_out rrule(::typeof(apply), ::LinOpAdjoint,::Any)