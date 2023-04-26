abstract type AbstractDomain{NTuple} end

struct CoordinateSpace{S} <: AbstractDomain{S} end

Base.size(::Type{<:AbstractDomain{S}}) where {S} = S
Base.size(::AbstractDomain{S}) where {S} = S
