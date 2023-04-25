module SimpleAlgebra
using ChainRulesCore

struct SimpleAlgebraFailure <: Exception
    msg::String
end
showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)


abstract type AbstractMap{T<:Number,I,O} end



Base.eltype(::Type{<:AbstractMap{T,I,O}}) where {T,I,O} = T


function Base.convert(::Type{T}, obj::M) where {T<:Number,T2,M<:AbstractMap{T2}} 
   return M.name.wrapper{T}(obj)
end

function Base.convert(::Type{T}, obj::M) where {T<:Number,M<:AbstractMap{T}} 
	return obj
end


 (A::AbstractMap)( v) = A*v
function Base.:*(A::AbstractMap, v) 
	
	return apply(A, v )
end 

Base.:*(A::AbstractMap, B::AbstractMap) = compose(A,B)

function compose(A::AbstractMap{T1,M,O}, B::AbstractMap{T2,I,M}) where {T1,T2,I,O,M}
 	T = promote_type(T1,T2)
	compose(convert(T,A),convert(T,B))
end


function compose(A::AbstractMap{T1,M,O}, B::AbstractMap{T2,I,N}) where {T1,T2,I,O,M,N}
	throw(SimpleAlgebraFailure("Input size of first element $M does not match the output of the second element $N"))
end



function compose(A::AbstractMap{T,M,O}, B::AbstractMap{T,I,M}) where {T,I,O,M}
	throw(SimpleAlgebraFailure("unimplemented operation"))
end

function Base.adjoint(A::AbstractMap{T,I,O}) where {T,I,O}
	throw(SimpleAlgebraFailure("unimplemented operation `adjoint` for mapping $(typeof(A))"))
end

function apply(A::AbstractMap,) 
	throw(SimpleAlgebraFailure("unimplemented operation `apply` for mapping $(typeof(A))"))

end

function apply_jacobian(A::AbstractMap,args... ) 
	throw(SimpleAlgebraFailure("unimplemented operation `apply_jacobian` for mapping $(typeof(A))"))
end



include("Cost/Cost.jl")
include("LinOp/LinOp.jl")
end # module SimpleAlgebra
