abstract type AbstractDomain{S} end

struct CoordinateSpace{S} <: AbstractDomain{S}  end #EuclidianSpace?

CoordinateSpace(sz::S) where {S<:NTuple} = CoordinateSpace{sz} 

Base.size(::Type{<:CoordinateSpace{S}}) where {S} = S
numel(::Type{CoordinateSpace{S}}) where {S} = prod(S)

Base.in(x::AbstractArray{T,N}, ::Type{S})  where {T,N,S<:CoordinateSpace} = size(S) == size(x)
Base.in(x::T, ::Type{S})  where {T,S<:CoordinateSpace} = size(S) == (1,)

Base.issubset(::Type{CoordinateSpace{S1}},::Type{CoordinateSpace{S2}}) where {S1,S2} = S1 ≤ S2
Base.issubset(::Type{D},::Type{D}) where {D<:CoordinateSpace} = true
Base.issubset(::Type{<:AbstractDomain},::Type{<:AbstractDomain}) = false
