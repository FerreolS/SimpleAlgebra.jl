

### ADJOINT ###

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)

struct LinOpAdjoint{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	LinOpAdjoint(A::AbstractLinOp{O,I}) where {I,O}   = new{I,O,typeof(A)}(A)
end

@functor LinOpAdjoint

LinOpAdjoint(A::LinOpAdjoint) = A.parent

inputspace(A::LinOpAdjoint)  = outputspace(A.parent)
outputspace(A::LinOpAdjoint) = inputspace(A.parent)


apply_(A::LinOpAdjoint, v) = apply_adjoint(A.parent,v)

apply_adjoint_(A::LinOpAdjoint, v) = apply(A.parent,v)

Base.adjoint(A::LinOpAdjoint) = A.parent	


### COMPOSITION ###

apply_adjoint_(A::MapComposition{I,O,L,R}, v) where {I,O,L<:AbstractLinOp,R<:AbstractLinOp} = apply_adjoint(A.right,apply_adjoint(A.left,v))

Base.adjoint(A::MapComposition{I,O,L,R}) where {I,O,L<:AbstractLinOp,R<:AbstractLinOp}   = A.right' * A.left'


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
