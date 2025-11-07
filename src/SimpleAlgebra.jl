module SimpleAlgebra

using ChainRulesCore
using Functors
using Adapt
using InverseFunctions
import InverseFunctions: inverse
using KernelAbstractions
using ArrayTools
using StaticArrays
using LinearAlgebra

export AbstractMap, AbstractCost
export AbstractDomain,
    CoordinateSpace,
    inputspace, outputspace,
    inputsize, outputsize,
    inputtype, outputtype
export MapReduceSum
export AbstractLinOp,
    LinOpConv,
    LinOpDFT,
    LinOpIdentity,
    LinOpScale,
    LinOpDiag,
    LinOpSelect,
    LinOpGrad,
    LinOpNFFT

export CostL2,
    CostHyperbolic


struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Utils/utils.jl")
include("Domain.jl")
include("Utils/Scratchspace.jl")
include("Map/Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")

end # module SimpleAlgebra
