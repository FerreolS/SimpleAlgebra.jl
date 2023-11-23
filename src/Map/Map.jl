abstract type AbstractMap{I<:AbstractDomain,O<:AbstractDomain} end

inputspace(A::AbstractMap)  = A.inputspace
outputspace(A::AbstractMap)  = A.outputspace
#inputspace(A::AbstractMap{I,I}) where {I} = A.inputspace

inputsize(A::AbstractMap)  = size(inputspace(A))
outputsize(A::AbstractMap) = size(outputspace(A))
#outputsize(A::AbstractMap{I,I}) where {I} = size(inputspace(A))

#= function Adapt.adapt_storage(::Type{T}, x::AbstractMap) where T
    fmap(adapt(T),x)
end
 =#

(A::AbstractMap)( v)   = A*v
#(A::Type{<:AbstractMap})(I::TI,O::TO,x...)  where {TI<:AbstractDomain,TO<:AbstractDomain} = A{I,O}(x...)

Base.:*(A::AbstractMap, v)  = apply(A, v ) 

Base.:*(A::AbstractMap, B::AbstractMap)   =  compose(A,B)

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


function compose(::AbstractMap, ::AbstractMap) 
	throw(SimpleAlgebraFailure("Input of first element does not match the output of the second element"))
end

# FIXME
#= function ChainRulesCore.rrule( ::typeof(apply_),A::AbstractMap, v)
    Map_pullback(Δy) = (NoTangent(),NoTangent(), apply_jacobian_(A, v,Δy))
    return  apply_(A,v), Map_pullback
end =#

include("./MapEmbedding.jl")
include("./OperationOnMap.jl")