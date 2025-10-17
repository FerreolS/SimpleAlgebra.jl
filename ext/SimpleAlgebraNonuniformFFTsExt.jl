module SimpleAlgebraNonuniformFFTsExt
using NonuniformFFTs, SimpleAlgebra
import SimpleAlgebra: LinOpNFFT, apply_, apply_adjoint_, apply_!, apply_adjoint_!


# Real-to-complex FFT.
function LinOpNFFT(
        ::Type{T},
        sz::NTuple{N, Int},
        points::NTuple{M, AbstractVector{T}};
        dims = 1:N,
        kwargs...
    ) where {T, N, M}

    if M == N
        dims === 1:N || error("When providing $M-dimensional points, dims must be 1:$M")
        plan_nufft = PlanNUFFT(T, sz; kwargs...)
        set_points!(plan_nufft, points)
        outputspace = CoordinateSpace(Complex{T}, size(plan_nufft))
        inputspace = CoordinateSpace(T, length(points[1]))
    else
    end

    return LinOpNFFT(inputspace, outputspace, plan_nufft)
end

apply_!(y, A::LinOpNFFT, x) = exec_type1!(y, A.plan, x)
apply_adjoint_!(y, A::LinOpNFFT, x) = exec_type2!(y, A.plan, x)

function apply_(A::LinOpNFFT, v)
    y = zeros(outputspace(A))
    apply_!(y, A, v)
    return y
end

function apply_adjoint_(A::LinOpNFFT, v)
    y = similar(v, inputspace(A))
    apply_adjoint_!(y, A, v)
    return y
end

end
