
# Steal the conversion mechanism from functor.jl in Flux

struct SimpleAlgebraEltypeAdaptor{T} end

Adapt.adapt_storage(::SimpleAlgebraEltypeAdaptor{T}, x::AbstractArray{<:Number}) where {T<:Number} = 
  convert(AbstractArray{T}, x)
Adapt.adapt_storage(::SimpleAlgebraEltypeAdaptor{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = 
  convert(AbstractArray{Complex{T}}, x)

_paramtype(::Type{T}, m) where T = fmap(adapt(SimpleAlgebraEltypeAdaptor{T}()), m)

# fastpath for arrays
_paramtype(::Type{T}, x::AbstractArray{<:Number}) where {T<:Number} = 
  convert(AbstractArray{T}, x)
_paramtype(::Type{T}, x::AbstractArray{<:Complex{<:AbstractFloat}}) where {T<:AbstractFloat} = 
  convert(AbstractArray{Complex{T}}, x)

"""
    f32(m)

Converts the `eltype` of model's *floating point* parameters to `Float32` (which is Flux's default).
Recurses into structs marked with [`@functor`](@ref).

See also [`f64`](@ref) and [`f16`](@ref).
"""
f32(m) = _paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's *floating point* parameters to `Float64`.
Recurses into structs marked with [`@functor`](@ref).

See also [`f32`](@ref) and [`f16`](@ref).
"""
f64(m) = _paramtype(Float64, m)

"""
    f16(m)

Converts the `eltype` of model's *floating point* parameters to `Float16`.
Recurses into structs marked with [`@functor`](@ref).

Support for `Float16` is limited on many CPUs. Julia may
convert to `Float32` for each operation, which is slow.

See also [`f32`](@ref) and [`f64`](@ref).

# Example
```jldoctest
julia> m = Chain(Dense(784, 2048, relu), Dense(2048, 10))  # all Float32
Chain(
  Dense(784 => 2048, relu),             # 1_607_680 parameters
  Dense(2048 => 10),                    # 20_490 parameters
)                   # Total: 4 arrays, 1_628_170 parameters, 6.211 MiB.

julia> m |> f16  # takes half the memory
Chain(
  Dense(784 => 2048, relu),             # 1_607_680 parameters
  Dense(2048 => 10),                    # 20_490 parameters
)                   # Total: 4 arrays, 1_628_170 parameters, 3.106 MiB.
```
"""
f16(m) = _paramtype(Float16, m)


Adapt.adapt_storage(::SimpleAlgebraEltypeAdaptor{T}, x::AbstractMap)  where T = Adapt.adapt_storage(T, x)
adapt_structure(to, M::AbstractMap) = fmap(to, M)
