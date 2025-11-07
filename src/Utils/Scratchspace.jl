struct Scratchspace{A}
    scratch::A
    function Scratchspace(scratch::M) where {M <: DenseVector}
        return new{typeof(scratch)}(scratch)
    end
end

function Scratchspace(::Type{M}) where {T, M <: DenseArray{T}}
    scratch = Base.typename(M).wrapper{T, 1}(undef, 1)
    return Scratchspace(scratch)
end
function Scratchspace(::Type{M}, ::T) where {T, M <: DenseArray}
    scratch = Base.typename(M).wrapper{T, 1}(undef, 1)
    return Scratchspace(scratch)
end

#= function newarray!(ssp::Scratchspace{A},csp::CoordinateSpace{T,N}) where {T,N,A<:AbstractVector{T}}
    len = length(csp)
	if length(ssp.scratch) < len
		resize!(ssp.scratch,len)
	end
	return reshape(view(ssp.scratch,1:len),size(csp))
end =#

function newarray!((; scratch)::Scratchspace{A}, csp::CoordinateSpace{T, N}) where {T, T2, N, A <: DenseVector{T2}}
    T3 = isconcretetype(T) ? T : T2
    bitlen = length(csp) * sizeof(T3)
    if sizeof(scratch) < bitlen
        len = div(bitlen, sizeof(T2), RoundUp)
        resize!(scratch, len)
    end
    #return reshape(view(reinterpret(T,ssp.scratch),1:length(csp)),size(csp))
    #  unsafe_wrap(Base.typename(A).wrapper, pointer(reinterpret(T3, scratch)), size(csp))
    return unsafe_wrap(Base.typename(A).wrapper, Ptr{T3}(pointer(scratch)), size(csp))
end

clear!(ssp::Scratchspace) = resize!(ssp.scratch, 1)

Base.resize!(ssp::Scratchspace, nl) = resize!(ssp.scratch, nl)

Adapt.adapt_structure(to, ssp::Scratchspace) = Scratchspace(adapt(to, ssp.scratch))

using LinearAlgebra


struct MySimilar
    arr::Vector{Float64}
    function MySimilar()
        return new(Float64[])
    end
end

function ((; arr)::MySimilar)(x::AbstractArray, ::Type{T}) where {T}
    isbitstype(T) || throw("MySimilar only supports isbits element types")
    bitlen = length(x) * sizeof(T)
    if sizeof(arr) < bitlen
        len = div(bitlen, sizeof(Float64), RoundUp)
        resize!(arr, len)
    end
    return unsafe_wrap(Array, Ptr{T}(pointer(arr)), size(x))
end

function similar2((; arr)::MySimilar, x::AbstractArray, ::Type{T}) where {T}
    isbitstype(T) || throw("MySimilar only supports isbits element types")
    bitlen = length(x) * sizeof(T)
    if sizeof(arr) < bitlen
        len = div(bitlen, sizeof(Float64), RoundUp)
        resize!(arr, len)
    end
    return reshape(view(reinterpret(T, arr), 1:length(x)), size(x))
end

clear!((; arr)::MySimilar) = resize!(arr, 0)

struct MyOperator{TA, TB}
    A::Matrix{TA}
    B::Matrix{TB}
    mysimilar::MySimilar
    function MyOperator(A::Matrix{TA}, B::Matrix{TB}) where {TA, TB}
        return new{TA, TB}(A, B, MySimilar())
    end
end

function apply_similar!(y::Vector{Ty}, (; A, B)::MyOperator, x::Vector{Tx}) where {Tx, Ty}
    z = similar(axes(B, 1), promote_type(Tx, eltype(B)))
    mul!(z, B, x)
    return mul!(y, A, z)
end

function apply_similar2!(y::Vector{Ty}, (; A, B, mysimilar)::MyOperator, x::Vector{Tx}) where {Tx, Ty}
    z = similar2(mysimilar, axes(B, 1), promote_type(Tx, eltype(B)))
    mul!(z, B, x)
    return mul!(y, A, z)
end

function apply!(y::Vector{Ty}, (; A, B, mysimilar)::MyOperator, x::Vector{Tx}) where {Tx, Ty}
    z = mysimilar(axes(B, 1), promote_type(Tx, eltype(B)))
    mul!(z, B, x)
    return mul!(y, A, z)
end


#=
julia> M = 5
5

julia> N = 100_000_000
100000000

julia> x  = randn(Float64,M);

julia> y = A * B * x;

julia> @benchmark $A * $B * $x
BenchmarkTools.Trial: 7 samples with 1 evaluation per sample.
 Range (min … max):  2.180 s … 25.368 s  ┊ GC (min … max): 4.08% … 1.32%
 Time  (median):     2.356 s             ┊ GC (median):    4.39%
 Time  (mean ± σ):   6.193 s ±  8.598 s  ┊ GC (mean ± σ):  1.91% ± 1.79%

  █                                                         
  █▁▁▁▁▁▁▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▁
  2.18 s        Histogram: frequency by time        25.4 s <

 Memory estimate: 1.49 GiB, allocs estimate: 7.

julia> apply!(y,Op,x);

julia> @benchmark apply!($y,$Op,$x)
BenchmarkTools.Trial: 13 samples with 1 evaluation per sample.
 Range (min … max):  1.960 s …    3.750 s  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.242 s               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.521 s ± 587.775 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █ █████  █ █           █     █  █                █       █  
  █▁█████▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁█ ▁
  1.96 s         Histogram: frequency by time         3.75 s <

 Memory estimate: 80 bytes, allocs estimate: 3.

julia> @benchmark apply_similar2!($y,$Op,$x)
BenchmarkTools.Trial: 10 samples with 1 evaluation per sample.
 Range (min … max):  1.790 s … 7.291 s  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     2.339 s            ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.419 s ± 1.969 s  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █▃                                                       
  ██▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▁▁▇▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇ ▁
  1.79 s        Histogram: frequency by time       7.29 s <

 Memory estimate: 0 bytes, allocs estimate: 0.

julia> @benchmark apply_similar!($y,$Op,$x)
BenchmarkTools.Trial: 13 samples with 1 evaluation per sample.
 Range (min … max):  2.329 s …    2.731 s  ┊ GC (min … max): 4.40% … 4.28%
 Time  (median):     2.498 s               ┊ GC (median):    4.08%
 Time  (mean ± σ):   2.469 s ± 129.368 ms  ┊ GC (mean ± σ):  4.16% ± 1.50%

  ██   ▁ ▁                ▁   ▁  ▁▁  █                     ▁  
  ██▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁█▁▁██▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  2.33 s         Histogram: frequency by time         2.73 s <

 Memory estimate: 1.49 GiB, allocs estimate: 3.

julia> 
=#
