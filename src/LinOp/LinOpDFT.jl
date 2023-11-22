
import AbstractFFTs: Plan, fftshift, ifftshift
using FFTW
import FFTW: fftwNumber, fftwReal, fftwComplex, FFTWPlan, cFFTWPlan, rFFTWPlan
import Adapt: adapt_storage
# All planning flags.
const PLANNING = (FFTW.ESTIMATE | FFTW.MEASURE | FFTW.PATIENT |
                  FFTW.EXHAUSTIVE | FFTW.WISDOM_ONLY)


struct LinOpDFT{I,O,
					F<:Plan,     # type of forward plan
					B<:Plan,      # type of backward plan
				} <:  AbstractLinOp{I,O}
    inputspace::I
    outputspace::O
    forward::F             # plan for forward transform
    backward::B            # plan for backward transform
    LinOpDFT(inputspace::I, outputspace::O, forward::F, backward::B) where {I,O,F,B} = new{I,O,F,B}(inputspace,outputspace,forward,backward)
	# unitary::Bool ?
end

@functor LinOpDFT

LinOpDFT(dims::NTuple;kwds...) = LinOpDFT(ComplexF64,dims; kwds...)

# Real-to-complex FFT.
function LinOpDFT(::Type{T},
                	dims::NTuple{N,Int};
                    timelimit::Real = FFTW.NO_TIMELIMIT,
                    flags::Integer = FFTW.MEASURE) where {T<:fftwReal,N}
    # Check arguments and build dimension list of the result of the forward
    # real-to-complex (r2c) transform.
    planning = check_flags(flags)

    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms implemented in FFTW, see
    # http://www.fftw.org/doc/Planner-Flags.html).
    forward = plan_rfft(Array{T}(undef, dims);
                        flags = (planning | FFTW.PRESERVE_INPUT),
                        timelimit = timelimit)
    backward = plan_brfft(Array{Complex{T}}(undef, forward.osz), dims[1];
                          flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)

    # Build operator.

	inputspace = CoordinateSpace(T,forward.sz)
	outputspace = CoordinateSpace(Complex{T},forward.osz)
    F = typeof(forward)
    B = typeof(backward)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end


# Complex-to-complex FFT.
function LinOpDFT(::Type{T},
                     dims::NTuple{N,Int};
                     timelimit::Real = FFTW.NO_TIMELIMIT,
                     flags::Integer = FFTW.MEASURE) where {T<:fftwComplex,N}
    # Check arguments.  The input and output of the complex-to-complex
    # transform have the same dimensions.
    planning = check_flags(flags)
    temp = Array{T}(undef, dims)

    # Compute the plans with suitable FFTW flags.  For maximum efficiency, the
    # transforms are always applied in-place and thus cannot preserve their
    # inputs.
    forward = plan_fft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                        timelimit = timelimit)
    backward = plan_bfft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)

    # Build operator.

	inputspace = CoordinateSpace(T,forward.sz)
	outputspace = CoordinateSpace(T,forward.osz)
    F = typeof(forward)
    B = typeof(backward)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end



# Constructor for dimensions not specified as a tuple.
LinOpDFT(T::Type{<:fftwNumber}, dims::Integer...; kwds...) =
    LinOpDFT(T, dims; kwds...)
# Constructor for transforms applicable to a given array.

apply_(A::LinOpDFT, v)  = A.forward * v
apply_adjoint_(A::LinOpDFT, v)  = A.backward * v

function compose(left::D, right::C)  where {I,O,F,B,C<:LinOpDFT{I,O,F,B},D<:LinOpAdjoint{O,I,C}} 
    if left.parent===right
		return LinOpScale(inputsize(right),length(inputspace(right)))
	end
	return LinOpComposition(left,right)
end

function ChainRulesCore.rrule( ::typeof(apply_),A::LinOpDFT, v)
    LinOpDFT_pullback(Δy) = (NoTangent(),NoTangent(), apply_adjoint_(A, Δy))
    return  apply_(A,v), LinOpDFT_pullback
end


#------------------------------------------------------------------------------
# Utilities borrowed from LazyAlgebra

"""

`check_flags(flags)` checks whether `flags` is an allowed bitwise-or
combination of FFTW planner flags (see
http://www.fftw.org/doc/Planner-Flags.html) and returns the filtered flags.

"""
function check_flags(flags::Integer)
    planning = flags & PLANNING
    flags == planning ||
        bad_argument("only FFTW planning flags can be specified")
    return UInt32(planning)
end

"""

`get_dimension(dims, i)` yields the `i`-th dimension in tuple of integers
`dims`.  Like for broadcasting rules, it is assumed that the length of
all dimensions after the last one are equal to 1.

"""
get_dimension(dims::NTuple{N,Int}, i::Integer) where {N} =
    (i < 1 ? bad_dimension_index() : i ≤ N ? dims[i] : 1)
# FIXME: should be in ArrayTools
bad_dimension_index() = error("invalid dimension index")


"""
```julia
goodfftdim(len)
```

yields the smallest integer which is greater or equal `len` and which is a
multiple of powers of 2, 3 and/or 5.  If argument is an array dimesion list
(i.e. a tuple of integers), a tuple of good FFT dimensions is returned.

Also see: [`goodfftdims`](@ref), [`rfftdims`](@ref).

"""
goodfftdim(len::Integer) = goodfftdim(Int(len))
goodfftdim(len::Int) = nextprod([2,3,5], len)

"""
```julia
goodfftdims(dims)
```

yields a list of dimensions suitable for computing the FFT of arrays whose
dimensions are `dims` (a tuple or a vector of integers).

Also see: [`goodfftdim`](@ref), [`rfftdims`](@ref).

"""
goodfftdims(dims::Integer...) = map(goodfftdim, dims)
goodfftdims(dims::Union{AbstractVector{<:Integer},Tuple{Vararg{Integer}}}) =
    map(goodfftdim, dims)

"""
```julia
rfftdims(dims)
```

yields the dimensions of the complex array produced by a real-complex FFT of a
real array of size `dims`.

Also see: [`goodfftdim`](@ref).

"""
rfftdims(dims::Integer...) = rfftdims(dims)
rfftdims(dims::NTuple{N,Integer}) where {N} =
    ntuple(d -> (d == 1 ? (Int(dims[d]) >>> 1) + 1 : Int(dims[d])), Val(N))
# Note: The above version is equivalent but much faster than
#     ((dims[1] >>> 1) + 1, dims[2:end]...)
# which is not optimized out by the compiler.

"""
### Generate Discrete Fourier Transform frequency indexes or frequencies

Syntax:

```julia
k = fftfreq(dim)
f = fftfreq(dim, step)
```

With a single argument, the function returns a vector of `dim` values set with
the frequency indexes:

```
k = [0, 1, 2, ..., n-1, -n, ..., -2, -1]   if dim = 2*n
k = [0, 1, 2, ..., n,   -n, ..., -2, -1]   if dim = 2*n + 1
```

depending whether `dim` is even or odd.  These rules are compatible to what is
assumed by `fftshift` (which to see) in the sense that:

```
fftshift(fftfreq(dim)) = [-n, ..., -2, -1, 0, 1, 2, ...]
```

With two arguments, `step` is the sample spacing in the direct space and the
result is a floating point vector with `dim` elements set with the frequency
bin centers in cycles per unit of the sample spacing (with zero at the start).
For instance, if the sample spacing is in seconds, then the frequency unit is
cycles/second.  This is equivalent to:

```
fftfreq(dim)/(dim*step)
```

See also: [`FFTOperator`](@ref), [`fftshift`](@ref).

"""
function fftfreq(_dim::Integer)
    dim = Int(_dim)
    n = div(dim, 2)
    f = Array{Int}(undef, dim)
    @inbounds begin
        for k in 1:dim-n
            f[k] = k - 1
        end
        for k in dim-n+1:dim
            f[k] = k - (1 + dim)
        end
    end
    return f
end

function fftfreq(_dim::Integer, step::Real)
    dim = Int(_dim)
    scl = Cdouble(1/(dim*step))
    n = div(dim, 2)
    f = Array{Cdouble}(undef, dim)
    @inbounds begin
        for k in 1:dim-n
            f[k] = (k - 1)*scl
        end
        for k in dim-n+1:dim
            f[k] = (k - (1 + dim))*scl
        end
    end
    return f
end