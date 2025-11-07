struct LinOp1DPolynomial{I, O, X <: Union{AbstractArray{<:AbstractFloat}, AbstractFloat}, P} <: AbstractLinOp{I, O}
    inputspace::I
    outputspace::O
    x::X
    precond::P
    LinOp1DPolynomial(inputspace::I, outputspace::O, x::X, p::P) where {I <: CoordinateSpace, O <: CoordinateSpace, X <: Union{AbstractArray, Number}, P} =
        new{I, O, X, P}(inputspace, outputspace, x, p)
end


LinOp1DPolynomial(x, order) = LinOp1DPolynomial(Number, x, order)

function LinOp1DPolynomial(::Type{TI}, x, degree) where {TI}
    inputspace = CoordinateSpace(TI, (degree + 1,))
    TO = isconcretetype(TI) ? typeof(oneunit(TI) * oneunit(eltype(x))) : TI
    outputspace = CoordinateSpace(TO, size(x))
    precond = [ sqrt(length(x) / sum(x .^ (2 * n))) for n in 0:degree]
    return LinOp1DPolynomial(inputspace, outputspace, x, precond)
end

function apply_(A::LinOp1DPolynomial, coefs)
    coefs .*= A.precond
    degree = length(A.inputspace) - 1
    y = similar(A.x)
    backend = get_backend(coefs)
    LinOp1DPolynomialKernel_(backend)(y, A.x, coefs, degree, ndrange = size(A.x))
    synchronize(backend)
    return y
    #return mapreduce( n -> coefs[n+1] .* (A.x).^n,+,0:degree) # no GPU compatible
    #return reduce(hcat,[ (A.x).^n  for n ∈ 0:degree])*coefs
end

@kernel function LinOp1DPolynomialKernel_(Y, X, coefs, degree)
    I = @index(Global, Cartesian)
    Y[I] = coefs[1]
    for n in 1:degree
        Y[I] += X[I] .^ n * coefs[n + 1]
    end
end


function apply_adjoint_(A::LinOp1DPolynomial{I, O, X}, x) where {I, O, X}
    T = isconcretetype(eltype(I)) ? eltype(I) : eltype(x)
    degree = length(A.inputspace) - 1
    y = similar(x, T, (degree + 1,))
    backend = get_backend(x)
    LinOp1DPolynomialKernel_adjoint_(backend)(y, A.x, x, ndrange = (degree + 1,))
    synchronize(backend)
    return y .* A.precond
end


@kernel function LinOp1DPolynomialKernel_adjoint_(Y, X, input)
    (I,) = @index(Global, NTuple)
    Y[I] = sum(X .^ (I - 1) .* input)
end
#= 

function ChainRulesCore.rrule( ::typeof(apply_),A::LinOp1DPolynomial, v)
	LinOp1DPolynomial_pullback(Δy) = (NoTangent(),NoTangent(), apply_adjoint_(A, Δy))
    return  apply_(A,v), LinOp1DPolynomial_pullback
end
 =#

struct LinOpPermutedims{I, O} <: AbstractLinOp{I, O}
    inputspace::I
    outputspace::O
    dims::Vector{Int}
    function LinOpPermutedims(inputspace::I, outputspace::O, dims::Vector{Int}) where {I <: CoordinateSpace, O <: CoordinateSpace}
        length(dims) != ndims(inputspace) && throw(ArgumentError("The length of dims must be equal to the number of dimensions of the input space"))
        length(unique(dims)) != length(dims) && throw(ArgumentError("The dims vector must not contain duplicate entries"))
        length(dims) != ndims(outputspace) && throw(ArgumentError("The length of dims must be equal to the number of dimensions of the output space"))
        return new{I, O}(inputspace, outputspace, dims)
    end
end

function LinOpPermutedims(::Type{TI}, sz::NTuple{N, Int}, dims::Vector{Int}) where {TI, N}

    inputspace = CoordinateSpace(TI, sz)

    outputspace = CoordinateSpace(TI, sz[dims])
    return LinOpPermutedims(inputspace, outputspace, dims)
end

LinOpPermutedims(sz::NTuple{N, Int}, dims::Vector{Int}) where {N} = LinOpPermutedims(Number, sz, dims)

function apply_(A::LinOpPermutedims, x)
    return permutedims(x, A.dims)
end
function apply_adjoint_(A::LinOpPermutedims, x)
    return permutedims(x, invperm(A.dims))
end

struct LinOpSlice{I, O, P, D} <: AbstractLinOp{I, O}
    inputspace::I
    outputspace::O
    linop::P
    dims::D
    function LinOpSlice(inputspace::I, outputspace::O, operator::P, dims::D) where {I <: CoordinateSpace, O <: CoordinateSpace, P, D}
        return new{I, O, P, D}(inputspace, outputspace, operator, dims)
    end
end

function LinOpSlice(sz::NTuple{N, Int}, operator::P, dims::Vector{Int}) where {N, P <: AbstractMap}
    inputspace = CoordinateSpace(inputtype(operator), sz)
    sz[dims] != inputsize(operator) && throw(ArgumentError("The size of the operator does not match the selected dimensions"))
    outputsz = (sz[1:(dims[1] - 1)]..., outputsize(operator)..., sz[(dims[end] + 1):length(sz)]...)
    outputspace = CoordinateSpace(outputtype(operator), outputsz)
    return LinOpSlice(inputspace, outputspace, operator, dims)
end

function apply_!(y, (; inputspace, outputspace, linop, dims)::LinOpSlice, x)
    inputsz = size(inputspace)
    ndrange = (inputsz[1:(dims[1] - 1)]..., inputsz[(dims[end] + 1):length(inputsz)]...)
    backend = get_backend(x)
    LinOpSlice_kernel(backend)(y, x, linop, dims[1], ndrange = ndrange)
    synchronize(backend)
    return y
end

function apply_(A::LinOpSlice, x)
    outputsz = size(outputspace(A))
    y = similar(x, outputsz)
    return apply_!(y, A, x)
end


@kernel function LinOpSlice_kernel(Y, X, A, dims)
    I = @index(Global, Cartesian)
    I1 = CartesianIndex(I.I[1:(dims - 1)])
    I2 = CartesianIndex(I.I[dims:end])
    #view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2) .= A * view(X, I1, colons(Val(ndims(inputspace(A))))..., I2)
    apply!(view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2), A, view(X, I1, colons(Val(ndims(inputspace(A))))..., I2))

    #Y[I1, .., I2] .= A * X[I1, .., I2]
end

struct LinOpNFFT{
        I,
        O,
        F,
    } <: AbstractLinOp{I, O}

    inputspace::I
    outputspace::O
    plan::F             # plan for forward transform
    dims::Vector{Int}  # dimensions along which the transform is applied

    LinOpNFFT(inputspace::I, outputspace::O, plan::F, dims) where {I <: CoordinateSpace, O <: CoordinateSpace, F} = new{I, O, F}(inputspace, outputspace, plan, dims)
end


function LinOpNFFT(args...; kwargs...)
    error("Load NonuniformFFTs.jl to use LinOpNFFT")
end
