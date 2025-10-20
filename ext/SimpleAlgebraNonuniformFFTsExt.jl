module SimpleAlgebraNonuniformFFTsExt
using NonuniformFFTs, SimpleAlgebra, ArrayTools
import SimpleAlgebra: LinOpNFFT, apply_, apply_adjoint_, apply_!, apply_adjoint_!


# Real-to-complex FFT.
function LinOpNFFT(
        ::Type{T},
        sz::NTuple{N, Int},
        points::NTuple{M, AbstractVector{T2}};
        dims = 1:N,
        kwargs...
    ) where {T1 <: Real, T <: Union{T1, Complex{T1}}, T2, N, M}

    if T1 != T2
        points = map(p -> convert.(T1, p), points)
    end
    if M == N
        dims == 1:N || error("When providing $M-dimensional points, dims must be 1:$M")
        plan_nufft = PlanNUFFT(T, sz; kwargs...)
        set_points!(plan_nufft, points)
        outputspace = CoordinateSpace(T <: Real ? Complex{T} : T, size(plan_nufft))
        inputspace = CoordinateSpace(T, length(points[1]))
    else
        dims == 1:length(dims) || error("When providing $M-dimensional points, dims must be 1:$M")
        t = trues(N)
        t[dims] .= false
        sum(t) + M == N || error("Length of dims must be N - $M when providing $M-dimensional points")
        ntrans = Val(prod(sz[t]))
        plan_nufft = PlanNUFFT(T, sz[dims]; ntransforms = ntrans, kwargs...)
        set_points!(plan_nufft, points)

        outputsize = ntuple(i -> t[i] ? sz[i] : size(plan_nufft)[findfirst(==(i), dims)], N)
        outputspace = CoordinateSpace(T <: Real ? Complex{T} : T, outputsize)
        inputsize = (length(points[1]), sz[t]...)
        #outputspace = CoordinateSpace(T <: Real ? Complex{T} : T, size(plan_nufft))
        inputspace = CoordinateSpace(T, inputsize)
    end

    return LinOpNFFT(inputspace, outputspace, plan_nufft, dims)
end


function apply_!(y, (; inputspace, outputspace, plan, dims)::LinOpNFFT{I, O, P}, x) where {T, N, M, I, O, P <: PlanNUFFT{T, N, M}}

    #=     if dims != collect(1:ndims(outputspace))
        x = reshape(permutedims(x, vcat(dims, setdiff(1:ndims(outputspace), dims))), size(outputspace)[dims]..., :)
    end =#

    sizein = size(inputspace)
    sizeout = size(outputspace)
    t = trues(length(sizeout))
    t[dims] .= false

    nx = ntuple(i -> view(reshape(x, sizein[1], :), :, i), M)
    ny = ntuple(i -> view(reshape(y, sizeout[.!t]..., :), colons(Val(N))..., i), M)
    return exec_type1!(ny, plan, nx)
end

function apply_adjoint_!(y, (; inputspace, outputspace, plan, dims)::LinOpNFFT{I, O, P}, x) where {T, N, M, I, O, P <: PlanNUFFT{T, N, M}}
    #= 
    if dims != collect(1:ndims(outputspace))
        x = reshape(permutedims(x, vcat(dims, setdiff(1:ndims(outputspace), dims))), size(outputspace)[dims]..., :)
    end =#

    sizein = size(inputspace)
    sizeout = size(outputspace)
    t = trues(length(sizeout))
    t[dims] .= false

    ny = ntuple(i -> view(reshape(y, sizein[1], :), :, i), M)
    nx = ntuple(i -> view(reshape(x, sizeout[.!t]..., :), colons(Val(N))..., i), M)


    return exec_type2!(ny, plan, nx)
end

function apply_!(y, (; plan)::LinOpNFFT{I, O, P}, x) where {T, N, I, O, P <: PlanNUFFT{T, N, 1}}
    return exec_type1!(y, plan, x)
end
function apply_adjoint_!(y, (; plan)::LinOpNFFT{I, O, P}, x) where {T, N, I, O, P <: PlanNUFFT{T, N, 1}}
    return exec_type2!(y, plan, x)
end

function apply_(A::LinOpNFFT, v)
    y = similar(v, outputspace(A))
    apply_!(y, A, v)
    return y
end

function apply_adjoint_(A::LinOpNFFT, v)
    y = similar(v, inputspace(A))
    apply_adjoint_!(y, A, v)
    return y
end

end
