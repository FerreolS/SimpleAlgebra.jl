abstract type AbstractDomain end

struct CoordinateSpace{T,N} <: AbstractDomain
	eltype::Type
	size::NTuple{N,Int}
	#CoordinateSpace{T,N}()
end #EuclidianSpace?


const AbstractCoordinateSpace{N} =  CoordinateSpace{T,N} where T


CoordinateSpace(sz::Int) = CoordinateSpace(Tuple(sz)) 
CoordinateSpace(sz::NTuple) = CoordinateSpace(Number,sz) 
CoordinateSpace(T::Type,sz::Int) = CoordinateSpace(T,Tuple(sz)) 
CoordinateSpace(T::Type,sz::NTuple{N}) where { N} = CoordinateSpace{T,N}(T,sz)
CoordinateSpace{T,0}() where T = CoordinateSpace(T,()) 

const Scalar{T} = CoordinateSpace{T,0} where T
Scalar() = Scalar{Number}()

Base.size(sp::S) where{S<:CoordinateSpace} = sp.size
Base.size(sp::AbstractCoordinateSpace{N}, d) where {N} = d::Integer <= N ? size(sp)[d] : 1

Base.length(sp::S) where{S<:CoordinateSpace} = prod(sp.size)
Base.ndims(::CoordinateSpace{T,N}) where {T,N} = N
Base.eltype(::CoordinateSpace{T,N}) where {T,N} = T

Base.in(x::AbstractArray{T,N}, sp::CoordinateSpace{TS,N})  where {T,TS,N} = (size(sp) == size(x)) && (T<:TS)
Base.in(::AbstractArray, ::CoordinateSpace)   = false
Base.in(::T, ::CoordinateSpace{TS,0})  where {TS,T} = T<:TS
Base.in(::T, ::CoordinateSpace)  where T = false

#= Base.issubset(sp1::CoordinateSpace{N1},sp2::CoordinateSpace{N2}) where {N1,N2} =  size(sp1) ≤ size(sp2)
Base.issubset(::Type{D},::Type{D}) where {D<:AbstractDomain} = true
Base.issubset(::Type{<:AbstractDomain},::Type{<:AbstractDomain}) = false
 =#