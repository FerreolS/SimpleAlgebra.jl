
using Zygote: pullback
abstract type AbstractMap{AbstractDomain,AbstractDomain} end

sizein(::AbstractMap{I,O}) where {I,O} = size(I)
sizeout(::AbstractMap{I,O}) where {I,O} = size(O)
spacein(::AbstractMap{I,O}) where {I,O} = I
spaceout(::AbstractMap{I,O}) where {I,O} = O


(A::AbstractMap)( v) = A*v

function Base.:*(A::AbstractMap, v) 
	return apply(A, v )
end 

Base.:*(A::AbstractMap, B::AbstractMap) = compose(A,B)


function compose(A::AbstractMap{M,O}, B::AbstractMap{I,N}) where {I,O,M,N}
	throw(SimpleAlgebraFailure("Input size of first element $M does not match the output of the second element $N"))
end

function compose(A::AbstractMap{M,O}, B::AbstractMap{I,M}) where {I,O,M}
	throw(SimpleAlgebraFailure("unimplemented operation"))
end

function Base.adjoint(A::AbstractMap{I,O}) where {I,O}
	throw(SimpleAlgebraFailure("unimplemented operation `adjoint` for mapping $(typeof(A))"))
end

function apply(A::AbstractMap,) 
	throw(SimpleAlgebraFailure("unimplemented operation `apply` for mapping $(typeof(A))"))

end

function apply_jacobian(A::AbstractMap,v,x ) 
	return pullback(x->A*x,v)[2](x)[1]
#	throw(SimpleAlgebraFailure("unimplemented operation `apply_jacobian` for mapping $(typeof(A))"))
end
