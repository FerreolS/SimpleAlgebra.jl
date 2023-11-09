abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputsize(::AbstractMap{I,O}) where {I,O} = size(I)
outputsize(::AbstractMap{I,O}) where {I,O} = size(O)
inputspace(::AbstractMap{I,O}) where {I,O} = I
outputspace(::AbstractMap{I,O}) where {I,O} = O

(A::AbstractMap)( v)   = A*v
#(A::Type{<:AbstractMap})(I::TI,O::TO,x...)  where {TI<:AbstractDomain,TO<:AbstractDomain} = A{I,O}(x...)

Base.:*(A::AbstractMap, v)  = apply(A, v ) 

Base.:*(A::AbstractMap, B::AbstractMap)   =  compose(A,B)

Base.:+(A::AbstractMap, B::AbstractMap)   =  Base.sum((A,B))

function apply(A::AbstractMap{I,O}, v) where {I,O}
	@assert v ∈ I "The input size must be  $(size(I))"
	apply_(A, v)
end
function apply_ end

function apply_jacobian(A::AbstractMap{I,O}, v,x) where {I,O}
	@assert v ∈ I "The size of the second parameter must be  $(size(I))"
	@assert x ∈ O "The size of the third parameter must be $(size(O))"
	if applicable(apply_jacobian_,A,v,x)
		return apply_jacobian_(A,v,x)
	else
		throw(SimpleAlgebraFailure("unimplemented operation `apply_jacobian_` for Mapping $(typeof(A))"))
	end
end

function apply_jacobian_ end


function compose(::AbstractMap{M,O}, ::AbstractMap{I,N}) where {I,O,M,N}
	throw(SimpleAlgebraFailure("Input size of first element $M does not match the output of the second element $N"))
end
