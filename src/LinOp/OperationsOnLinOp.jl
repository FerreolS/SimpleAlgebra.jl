

### ADJOINT ###

Base.adjoint(A::AbstractLinOp) = AdjointLinOp(A)

struct AdjointLinOp{I,O,D<:AbstractLinOp} <:  AbstractLinOp{I,O}
	parent::D
	AdjointLinOp(A::AbstractLinOp{O,I}) where {I,O}   = new{I,O,typeof(A)}(A)
end

@functor AdjointLinOp

AdjointLinOp(A::AdjointLinOp) = A.parent

inputspace(A::AdjointLinOp)  = outputspace(A.parent)
outputspace(A::AdjointLinOp) = inputspace(A.parent)


apply_(A::AdjointLinOp, v) = apply_adjoint(A.parent,v)

apply_adjoint_(A::AdjointLinOp, v) = apply(A.parent,v)

Base.adjoint(A::AdjointLinOp) = A.parent	


### COMPOSITION ###

apply_adjoint_(A::CompositionMap{I,O,L,R}, v) where {I,O,L<:AbstractLinOp,R<:AbstractLinOp} = apply_adjoint(A.right,apply_adjoint(A.left,v))

Base.adjoint(A::CompositionMap{I,O,L,R}) where {I,O,L<:AbstractLinOp,R<:AbstractLinOp}   = A.right' * A.left'


### SUM ###

struct SumLinOp{I,O,D1<:AbstractLinOp{I,O},D2<:AbstractLinOp{I,O}} <:  AbstractLinOp{I,O}
	left::D1
	right::D2
	function SumLinOp(A::D1, B::D2) where {I,O, D1<:AbstractLinOp{I,O},  D2<:AbstractLinOp{I,O}} 
		    return new{O,I,D1,D2}(A,B)
	end
end

@functor SumLinOp

apply_(A::SumLinOp, x) = apply(A.left,x) .+ apply(A.right,x)

apply_adjoint_(A::SumLinOp, v) = apply_adjoint(A.right,v) .+ apply_adjoint(A.left,v)

Base.adjoint(A::SumLinOp)  = A.right' + A.left'
