

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)

compose(A::AbstractLinOp,B::AbstractLinOp) = LinOpComposition(A, B)
Base.sum(A::AbstractLinOp,B::AbstractLinOp) = LinOpSum(A, B)

### ADJOINT ###

struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{O,I,typeof(A)}(A)
end

@functor LinOpAdjoint

LinOpAdjoint(A::LinOpAdjoint) = A.parent

inputspace(A::LinOpAdjoint)  = outputspace(A.parent)
outputspace(A::LinOpAdjoint) = inputspace(A.parent)



function compose(A::LinOpAdjoint{I,O,D},B::D) where{I,O,D<:AbstractLinOp} 
	A.parent===B && return makeHtH(B)
	#throw(SimpleAlgebraFailure("unimplemented operation"))
	return LinOpComposition(A,B)
end

apply_(A::LinOpAdjoint, v) = apply_adjoint(A.parent,v)

apply_adjoint_(A::LinOpAdjoint, v) = apply(A.parent,v)

Base.adjoint(A::LinOpAdjoint) = A.parent	


### COMPOSITION ###

struct LinOpComposition{I,O,Dleft<:AbstractLinOp,Dright<:AbstractLinOp} <:  AbstractLinOp{I,O}
	left::Dleft
	right::Dright
	function LinOpComposition(A::Dleft, B::Dright) where {I1,O1, I2, O2,Dleft<:AbstractLinOp{I1,O1},  Dright<:AbstractLinOp{I2,O2}} 
		    return new{I2,O1, Dleft,Dright}(A,B)
	end
end

@functor LinOpComposition


inputspace(A::LinOpComposition)  = inputspace(A.right)
outputspace(A::LinOpComposition) = outputspace(A.left)


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

@functor LinOpSum

apply_(A::LinOpSum, x) = apply(A.left,x) .+ apply(A.right,x)

apply_adjoint_(A::LinOpSum, v) = apply_adjoint(A.right,v) .+ apply_adjoint(A.left,v)

Base.adjoint(A::LinOpSum)  = A.right' + A.left'
