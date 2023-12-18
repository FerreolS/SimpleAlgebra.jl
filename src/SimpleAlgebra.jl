module SimpleAlgebra

using ChainRulesCore
using Requires
using Functors
using Adapt
using InverseFunctions
import InverseFunctions:inverse
using  LoopVectorization
using  KernelAbstractions


export  AbstractMap,AbstractCost
export  AbstractDomain,
        CoordinateSpace,AbstractCoordinateSpace,
        inputspace, outputspace,
        inputsize, outputsize
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
include("Map/Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")
include("Utils/functor.jl")
#include("Utils/traits.jl")
function __init__()
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("Ext/ExtZygote.jl")
	@require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" include("Ext/ExtFlux.jl")
	@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("Ext/ExtCUDA.jl")
        @require KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c" include("Ext/ExtKernelAbtraction.jl")
    end
end

end # module SimpleAlgebra
