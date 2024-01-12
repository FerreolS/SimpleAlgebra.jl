abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputspace(A::AbstractMap)  = A.inputspace
outputspace(A::AbstractMap)  = A.outputspace

inputsize(A::AbstractMap)  = size(inputspace(A))
outputsize(A::AbstractMap) = size(outputspace(A))

inputtype(A::AbstractMap)  = eltype(inputspace(A))
outputtype(A::AbstractMap) = eltype(outputspace(A))

(A::AbstractMap)( v)   = A*v


Base.:+(A::AbstractMap, B::AbstractMap)  = add(A,B)
Base.:+(A::AbstractMap, B)= add(A,B)
Base.:+(B,A::AbstractMap) = add(A,B)
Base.:-(A::AbstractMap, B::AbstractMap)  = add(A,-B)
Base.:-(A::AbstractMap, B) = add(A,-B)
Base.:-(A,B::AbstractMap) = add(A,-B)
Base.:-(A::AbstractMap) = -1*A

Base.:*(A::AbstractMap, v)  = apply(A, v ) 

Base.:*(A::AbstractMap, B::AbstractMap)   =  compose(A,B)
Base.:∘(A::AbstractMap, B::AbstractMap)   =  compose(A,B)


Base.inv(A::AbstractMap) = inverse(A)

function Base.:/(A::T,B::AbstractMap) where T
	if A===B
		return LinOpIdentity(inputspace(B))
	end
    A * inv(B)
end
function Base.:\(B::AbstractMap,A::T) where T
	if A===B
		return LinOpIdentity(inputspace(B))
	end
    inv(B) * A
end

function Base.:/(A::AbstractMap,B::Number) 
    A * inv(B)
end
function Base.:\(A::Number,B::AbstractMap) 
    inv(A) * B
end

function apply(A::AbstractMap{I,O}, v) where {I,O}
	v ∈ inputspace(A) || throw(SimpleAlgebraFailure("The input size must belong to the space $(inputspace(A))")) 
	apply_(A, v)
end
function apply_ end

function apply_jacobian(A::AbstractMap{I,O}, v,x) where {I,O}
	 v ∈ I  || throw(SimpleAlgebraFailure("The size of the second parameter must be  $(size(I))"))
	 x ∈ O  || throw(SimpleAlgebraFailure("The size of the third parameter must be $(size(O))"))
	if applicable(apply_jacobian_,A,v,x)
		return apply_jacobian_(A,v,x)
	else
		throw(SimpleAlgebraFailure("unimplemented operation `apply_jacobian_` for Mapping $(typeof(A))"))
	end
end

function apply_jacobian_ end


Adapt.adapt_storage(to, x::AbstractMap) = fmap(adapt(to), x)

Adapt.adapt_storage(::Type{T}, x::AbstractMap)  where T<:Number = adapt(AbstractArray{T}, x)

#= function compose(::AbstractMap, ::AbstractMap) 
	throw(SimpleAlgebraFailure("Input of first element does not match the output of the second element"))
end =#

# FIXME
#= function ChainRulesCore.rrule( ::typeof(apply_),A::AbstractMap, v)
    Map_pullback(Δy) = (NoTangent(),NoTangent(), apply_jacobian_(A, v,Δy))
    return  apply_(A,v), Map_pullback
end =#

include("./OperationOnMap.jl")
include("./MapReduceSum.jl")