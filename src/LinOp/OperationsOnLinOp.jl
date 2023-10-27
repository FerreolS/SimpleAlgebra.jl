

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)


compose(A::AbstractLinOp,B::AbstractLinOp) = LinOpComposition(A, B)
sum(A::AbstractLinOp,B::AbstractLinOp) = LinOpSum(A, B)

### ADJOINT ###

struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{O,I,typeof(A)}(A)
end

function compose(A::LinOpAdjoint{I,O,D},B::D) where{I,O,D<:AbstractLinOp} 
	A.parent===B && return makeHtH(B)
	#throw(SimpleAlgebraFailure("unimplemented operation"))
	return LinOpComposition(A,B)
end

apply_(A::LinOpAdjoint, v) = apply_adjoint(A.parent,v)

apply_adjoint_(A::LinOpAdjoint, v) = apply(A.parent,v)

Base.adjoint(A::LinOpAdjoint) = A.parent	


### COMPOSITION ###

struct LinOpComposition{I,O,D1<:AbstractLinOp,D2<:AbstractLinOp} <:  AbstractLinOp{I,O}
	left::D1
	right::D2
	function LinOpComposition(A::D1, B::D2) where {I,O,IO, D1<:AbstractLinOp{IO,O},  D2<:AbstractLinOp{I,IO}} 
		    return new{O,I,D1,D2}(A,B)
	end
end
apply_(A::LinOpComposition, v) = apply(A.left,apply(A.right,v))

apply_adjoint_(A::LinOpComposition, v) = apply_adjoint(A.right,apply_adjoint(A.left,v))

Base.adjoint(A::LinOpComposition)  = A.right' * A.left'


### SUM ###

struct LinOpSum{I,O,D1<:AbstractLinOp{I,O},D2<:AbstractLinOp{I,O}} <:  AbstractLinOp{I,O}
	left::D1
	right::D2
	function LinOpSum(A::D1, B::D2) where {I,O, D1<:AbstractLinOp{I,O},  D2<:AbstractLinOp{I,O}} 
		    return new{O,I,D1,D2}(A,B)
	end
end
apply_(A::LinOpSum, x) = apply(A.left,x) .+ apply(A.right,x)

apply_adjoint_(A::LinOpSum, v) = apply_adjoint(A.right,v) .+ apply_adjoint(A.left,v)

Base.adjoint(A::LinOpSum)  = A.right' + A.left'
