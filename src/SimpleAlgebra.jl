module SimpleAlgebra

using ChainRulesCore
using Requires
using Functors
using Adapt
using InverseFunctions
import InverseFunctions:inverse

export  AbstractMap,AbstractCost
export  AbstractDomain,
        CoordinateSpace,AbstractCoordinateSpace,
        inputspace, outputspace,
        inputsize, outputsize
export  AbstractLinOp, 
        LinOpConv, 
        LinOpDFT,
        LinOpIdentity,
        LinOpScale,
        LinOpDiag,
        LinOpSelect
export CostL2


struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Domain.jl")
include("Map/Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")
include("Utils/functor.jl")
#include("Utils/traits.jl")
function __init__()
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("Ext/ExtZygote.jl")
	@require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" include("Ext/ExtFlux.jl")
	@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("Ext/ExtCUDA.jl")
end

end # module SimpleAlgebra
