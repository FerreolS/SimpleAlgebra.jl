abstract type AbstractDomain end

struct CoordinateSpace{T, N} <: AbstractDomain
    size::NTuple{N, Int}
end #EuclidianSpace?

#const AbstractCoordinateSpace{N} =  CoordinateSpace{T,N} where T


CoordinateSpace(sz::Int) = CoordinateSpace(Tuple(sz))
CoordinateSpace(sz::NTuple) = CoordinateSpace(Number, sz)
CoordinateSpace(T::Type, sz::Int) = CoordinateSpace{T, 1}(Tuple(sz))
CoordinateSpace(T::Type, sz::NTuple{N}) where {N} = CoordinateSpace{T, N}(sz)
CoordinateSpace{T, 0}() where {T} = CoordinateSpace(T, ())
CoordinateSpace(::Type{T}, sp::CoordinateSpace) where {T} = CoordinateSpace(T, sp.size)
CoordinateSpace(::Type{T}, sp::CoordinateSpace{T, N}) where {T, N} = sp
const Scalar{T} = CoordinateSpace{T, 0} where {T}
Scalar() = Scalar{Number}()

Base.size(sp::S) where {S <: CoordinateSpace} = sp.size
Base.size(sp::CoordinateSpace{T, N}, d) where {T, N} = d::Integer <= N ? size(sp)[d] : 1
#bytesize(sp::S) where {T,N,S<:CoordinateSpace{T,N}} = length(sp) * sizeof(T)
function Base.axes(A::CoordinateSpace{T, N}, d) where {T, N}
    @inline
    return d::Integer <= N ? axes(A)[d] : Base.OneTo(1)
end 


Base.length(sp::S) where {S <: CoordinateSpace} = prod(sp.size)
Base.ndims(::CoordinateSpace{T, N}) where {T, N} = N
Base.ndims(::Type{CoordinateSpace{T, N}}) where {T, N} = N
Base.eltype(::CoordinateSpace{T, N}) where {T, N} = T
Base.eltype(::Type{CoordinateSpace{T, N}}) where {T, N} = T
hasconcretetype(::Type{CoordinateSpace{T, N}}) where {T, N} = isconcretetype(T)
hasconcretetype(::CoordinateSpace{T, N}) where {T, N} = isconcretetype(T)

Base.in(x::AbstractArray{T, N}, sp::CoordinateSpace{TS, N}) where {TS, T <: TS, N} = (size(sp) == size(x))
Base.in(x::AbstractArray{T, N}, sp::CoordinateSpace{TS, N}) where {T <: AbstractFloat, TS <: Complex{T}, N} = (size(sp) == size(x))
Base.in(::AbstractArray, ::CoordinateSpace) = false
Base.in(::T, ::CoordinateSpace{TS, 0}) where {TS, T} = T <: TS
Base.in(::T, ::CoordinateSpace) where {T} = false

#= Base.issubset(sp1::CoordinateSpace{N1},sp2::CoordinateSpace{N2}) where {N1,N2} =  size(sp1) ≤ size(sp2)
Base.issubset(::Type{D},::Type{D}) where {D<:AbstractDomain} = true
Base.issubset(::Type{<:AbstractDomain},::Type{<:AbstractDomain}) = false
 =#

Base.zeros(sp::CoordinateSpace{T, N}) where {T, N} = zeros(T, size(sp))
Base.ones(sp::CoordinateSpace{T, N}) where {T, N} = zeros(T, size(sp))
Base.zeros(::Type{T}, sp::CoordinateSpace) where {T} = zeros(T, size(sp))
Base.ones(::Type{T}, sp::CoordinateSpace) where {T} = zeros(T, size(sp))
Base.rand(sp::CoordinateSpace{T, N}) where {T, N} = rand(T, size(sp))
Base.rand(::Type{T}, sp::CoordinateSpace) where {T} = rand(T, size(sp))
Base.randn(sp::CoordinateSpace{T, N}) where {T, N} = rand(T, size(sp))
Base.randn(::Type{T}, sp::CoordinateSpace) where {T} = rand(T, size(sp))

Base.similar(A::AbstractArray, sp::CoordinateSpace{T, N}) where {T, N} = isconcretetype(T) ? similar(A, T, size(sp)) : similar(A, size(sp))

Base.cat(sp1::CoordinateSpace{T, N}, sp2::CoordinateSpace{T, M}) where {T, N, M} = CoordinateSpace(T, (sp1.size..., sp2.size...))
Base.cat(sp1::CoordinateSpace{T, N}, sp2::NTuple) where {T, N} = CoordinateSpace(T, (sp1.size..., sp2...))
Base.cat(sp1::NTuple, sp2::CoordinateSpace{T, N}) where {T, N} = CoordinateSpace(T, (sp1..., sp2.size...))

Base.cat_size(A::CoordinateSpace) = size(A)

promote_space(args...) =
    CoordinateSpace(promote_type(map(eltype, args)...), map(x -> size(x)[1], Broadcast.combine_axes(args...)))

Adapt.adapt_storage(::Type{A}, x::CoordinateSpace{T, N}) where {T, N, A <: AbstractArray{T}} = x
Adapt.adapt_storage(::Type{A}, x::CoordinateSpace{Tx, N}) where {T, Tx, N, A <: AbstractArray{T}} = isconcretetype(Tx) ? CoordinateSpace(T, x) : x
Adapt.adapt_storage(::Type{T}, x::CoordinateSpace) where {T <: Number} = CoordinateSpace(T, x)
Adapt.adapt_storage(::Type{T}, x::CoordinateSpace{C, N}) where {N, T <: Real, C <: Complex} = CoordinateSpace(Complex{T}, x)
Adapt.adapt_storage(::Type{A}, x::CoordinateSpace{C, N}) where {N, T <: Real, C <: Complex, A <: AbstractArray{T}} = CoordinateSpace(Complex{T}, x)
Adapt.adapt_storage(::Type{T}, x::CoordinateSpace{C, N}) where {N, T <: Complex, C <: Complex} = CoordinateSpace(T, x)
Adapt.adapt_storage(::Type{A}, x::CoordinateSpace{C, N}) where {N, T <: Complex, C <: Complex, A <: AbstractArray{T}} = CoordinateSpace(T, x)
