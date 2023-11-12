abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputspace(A::AbstractMap{I,O}) where {I,O} = A.inputspace
outputspace(A::AbstractMap{I,O}) where {I,O} = A.outputspace
#inputspace(A::AbstractMap{I,I}) where {I} = A.inputspace

inputsize(A::AbstractMap{I,O}) where {I,O} = size(inputspace(A))
outputsize(A::AbstractMap{I,O}) where {I,O} = size(outputspace(A))
#outputsize(A::AbstractMap{I,I}) where {I} = size(inputspace(A))


(A::AbstractMap)( v)   = A*v
#(A::Type{<:AbstractMap})(I::TI,O::TO,x...)  where {TI<:AbstractDomain,TO<:AbstractDomain} = A{I,O}(x...)

Base.:*(A::AbstractMap, v)  = apply(A, v ) 

Base.:*(A::AbstractMap, B::AbstractMap)   =  compose(A,B)

Base.:+(A::AbstractMap, B::AbstractMap)   =  Base.sum((A,B))

function apply(A::AbstractMap{I,O}, v) where {I,O}
	@assert v ∈ inputspace(A) "The input size must belong to the space $(inputspace(A))"
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
