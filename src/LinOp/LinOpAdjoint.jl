struct LinOpAdjoint{T,I,O,D<:AbstractLinOp} <:  AbstractLinOp{T,I,O}
	Op::D
	LinOpAdjoint(A::AbstractLinOp{T,O,I}) where {T,I,O}   = new{T,O,I,typeof(A)}(A)
end

function LinOpAdjoint(A::LinOpAdjoint) 
	return A
end

LinOpAdjoint{T}(obj::LinOpAdjoint) where {T} = LinOpAdjoint(convert(T,obj.Op))

function compose(A::LinOpAdjoint{T,I,O,D},B::D) where{T,I,O,D<:AbstractLinOp} 
	if A.Op===B
		return makeHtH(B)
	else
		throw(SimpleAlgebraFailure("unimplemented operation"))
	end
end


function apply(A::LinOpAdjoint, v) 
	return apply_adjoint(A.Op,v)
end
