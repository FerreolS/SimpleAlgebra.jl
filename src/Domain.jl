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
CoordinateSpace(::Type{T},sp::CoordinateSpace) where T = CoordinateSpace(T,sp.size)
CoordinateSpace(::Type{T},sp::CoordinateSpace{T,N}) where {T,N} = sp
const Scalar{T} = CoordinateSpace{T,0} where T
Scalar() = Scalar{Number}()

Base.size(sp::S) where{S<:CoordinateSpace} = sp.size
Base.size(sp::AbstractCoordinateSpace{N}, d) where {N} = d::Integer <= N ? size(sp)[d] : 1
#bytesize(sp::S) where {T,N,S<:CoordinateSpace{T,N}} = length(sp) * sizeof(T)

Base.length(sp::S) where{S<:CoordinateSpace} = prod(sp.size)
Base.ndims(::CoordinateSpace{T,N}) where {T,N} = N
Base.ndims(::Type{CoordinateSpace{T,N}}) where {T,N} = N
Base.eltype(::CoordinateSpace{T,N}) where {T,N} = T
Base.eltype(::Type{CoordinateSpace{T,N}}) where {T,N} = T
hasconcretetype(::Type{CoordinateSpace{T,N}}) where {T,N} = isconcretetype(T)
hasconcretetype(::CoordinateSpace{T,N}) where {T,N} = isconcretetype(T)

Base.in(x::AbstractArray{T,N}, sp::CoordinateSpace{TS,N})  where {TS,T<:TS,N} = (size(sp) == size(x))
Base.in(x::AbstractArray{T,N}, sp::CoordinateSpace{TS,N})  where {T<:AbstractFloat,TS<:Complex{T},N} = (size(sp) == size(x))
Base.in(::AbstractArray, ::CoordinateSpace)   = false
Base.in(::T, ::CoordinateSpace{TS,0})  where {TS,T} = T<:TS
Base.in(::T, ::CoordinateSpace)  where T = false

#= Base.issubset(sp1::CoordinateSpace{N1},sp2::CoordinateSpace{N2}) where {N1,N2} =  size(sp1) ≤ size(sp2)
Base.issubset(::Type{D},::Type{D}) where {D<:AbstractDomain} = true
Base.issubset(::Type{<:AbstractDomain},::Type{<:AbstractDomain}) = false
 =#

Base.zeros(sp::CoordinateSpace{T,N}) where {T,N} = zeros(T,size(sp))
Base.ones(sp::CoordinateSpace{T,N}) where {T,N} = zeros(T,size(sp))
Base.zeros(::Type{T},sp::CoordinateSpace{N}) where {T,N} = zeros(T,size(sp))
Base.ones(::Type{T},sp::CoordinateSpace{N}) where {T,N} = zeros(T,size(sp))

Base.similar(A::AbstractArray, sp::CoordinateSpace{T,N}) where {T,N} = isconcretetype(T) ? similar(A, T, size(sp)) : similar(A,  size(sp))

promote_space(args...)  = 
 	CoordinateSpace(promote_type(map(eltype,args)...),map(x -> size(x)[1], Broadcast.combine_axes(args...)) )