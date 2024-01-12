module SimpleAlgebra

using ChainRulesCore
using Requires
using Functors
using Adapt
using InverseFunctions
import InverseFunctions:inverse
using  KernelAbstractions


export  AbstractMap,AbstractCost
export  AbstractDomain,
        CoordinateSpace,AbstractCoordinateSpace,
        inputspace, outputspace,
        inputsize, outputsize,
        inputtype, outputtype
export  MapReduceSum
export  AbstractLinOp, 
        LinOpConv, 
        LinOpDFT,
        LinOpIdentity,
        LinOpScale,
        LinOpDiag,
        LinOpSelect,
        LinOpGrad
export CostL2,
        CostHyperbolic


struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Domain.jl")
include("Utils/Scratchspace.jl")
include("Map/Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end # module SimpleAlgebra
