abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputspace(A::AbstractMap)  = A.inputspace
outputspace(A::AbstractMap)  = A.outputspace

inputsize(A::AbstractMap)  = size(inputspace(A))
outputsize(A::AbstractMap) = size(outputspace(A))

(A::AbstractMap)( v)   = A*v


Base.:+(A::AbstractMap, B::AbstractMap)  = add(A,B)
Base.:+(A::AbstractMap, B::T) where {T<:Union{Number,AbstractArray}}= add(A,B)
Base.:+(B::T,A::AbstractMap) where {T<:Union{Number,AbstractArray}} = add(A,B)


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



function apply(A::AbstractMap{I,O}, v) where {I,O}
	#@assert v ∈ inputspace(A) "The input size must belong to the space $(inputspace(A))"
	apply_(A, v)
end
function apply_ end

function apply_jacobian(A::AbstractMap{I,O}, v,x) where {I,O}
	#@assert v ∈ I "The size of the second parameter must be  $(size(I))"
	#@assert x ∈ O "The size of the third parameter must be $(size(O))"
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