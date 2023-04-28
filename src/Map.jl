
using Zygote: pullback
abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputsize(::AbstractMap{I,O}) where {I,O} = size(I)
outputsize(::AbstractMap{I,O}) where {I,O} = size(O)
inputspace(::AbstractMap{I,O}) where {I,O} = I
outputspace(::AbstractMap{I,O}) where {I,O} = O


(A::AbstractMap{I,O})( v)  where {I,O} = A*v

Base.:*(A::AbstractMap{I,O}, v)  where {I,O} = apply(A, v ) 

Base.:*(A::AbstractMap{M,O}, B::AbstractMap{I,N})  where {I,O,M,N} =  compose(A,B)

function compose(A::AbstractMap{M,O}, B::AbstractMap{I,N}) where {I,O,M,N}
	throw(SimpleAlgebraFailure("Input size of first element $M does not match the output of the second element $N"))
end

# function compose(A::AbstractMap{M,O}, B::AbstractMap{I,M}) where {I,O,M}
# 	throw(SimpleAlgebraFailure("unimplemented operation"))
# end

function apply_jacobian(A::AbstractMap{I,O},v,x )  where {I,O}
	@assert v ∈ I "The size of the second parameter must be  $(size(I))"
	@assert x ∈ O "The size of the third parameter must be $(size(O))"
	return pullback(x->A*x,v)[2](x)[1]
#	throw(SimpleAlgebraFailure("unimplemented operation `apply_jacobian` for mapping $(typeof(A))"))
end
