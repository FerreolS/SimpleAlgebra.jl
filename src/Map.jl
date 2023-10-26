
abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputsize(::AbstractMap{I,O}) where {I,O} = size(I)
outputsize(::AbstractMap{I,O}) where {I,O} = size(O)
inputspace(::AbstractMap{I,O}) where {I,O} = I
outputspace(::AbstractMap{I,O}) where {I,O} = O


(A::AbstractMap{I,O})( v)  where {I,O} = A*v

Base.:*(A::AbstractMap{I,O}, v)  where {I,O} = apply(A, v ) 

Base.:*(A::AbstractMap{M,O}, B::AbstractMap{I,N})  where {I,O,M,N} =  compose(A,B)

function compose(::AbstractMap{M,O}, ::AbstractMap{I,N}) where {I,O,M,N}
	throw(SimpleAlgebraFailure("Input size of first element $M does not match the output of the second element $N"))
end

# function compose(A::AbstractMap{M,O}, B::AbstractMap{I,M}) where {I,O,M}
# 	throw(SimpleAlgebraFailure("unimplemented operation"))
# end