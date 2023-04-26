struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	Op::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{O,I,typeof(A)}(A)
end

function LinOpAdjoint(A::LinOpAdjoint) 
	return A
end

function compose(A::LinOpAdjoint{I,O,D},B::D) where{I,O,D<:AbstractLinOp} 
	if A.Op===B
		return makeHtH(B)
	else
		throw(SimpleAlgebraFailure("unimplemented operation"))
	end
end


function apply(A::LinOpAdjoint, v) 
	return apply_adjoint(A.Op,v)
end

function apply_adjoint(A::LinOpAdjoint, v) 
	return apply(A.Op,v)
end

# #@opt_out rrule(::typeof(apply), ::LinOpAdjoint,::Any)