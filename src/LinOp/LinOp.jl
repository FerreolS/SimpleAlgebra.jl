
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end

apply_adjoint(A::AbstractLinOp,x) = apply_jacobian(A,zeros(eltype(x),inputsize(A)),x)

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)

compose(A::AbstractLinOp,B::AbstractLinOp) = LinOpComposition(A, B)

# FIXME issue here should be generated only when apply_adjoint is implemented
function ChainRulesCore.rrule( ::typeof(apply),A::AbstractLinOp, v)
    ∂Y(Δy) = (NoTangent(),NoTangent(), apply_adjoint(A,Δy))
    return apply(A,v), ∂Y
end

include("./LinOpComposition.jl")
include("./LinOpDiag.jl")
include("./LinOpAdjoint.jl")
include("./LinOpDFT.jl")
include("./LinOpConv.jl")
