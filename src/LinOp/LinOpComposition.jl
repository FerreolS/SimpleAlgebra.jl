struct LinOpComposition{I,O,D1<:AbstractLinOp,D2<:AbstractLinOp} <:  AbstractLinOp{I,O}
	left::D1
	right::D2
	function LinOpCompose(A::D1, B::D2) where {I,O,IO, D1<:AbstractLinOp{IO,O},  D2<:AbstractLinOp{I,IO}} 
		    return new{O,I,D1,D2}(A,B)
	end
end
apply(A::LinOpComposition, v) = apply(A.left,apply(A.right,v))

apply_adjoint(A::LinOpComposition, v) = apply_adjoint(A.right,apply_adjoint(A.left,v))

Base.adjoint(A::LinOpComposition)  = A.right' * A.left'
