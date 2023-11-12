module SimpleAlgebra

using ChainRulesCore
using Requires
using Functors

struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Domain.jl")
include("Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")

function __init__()
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("Utils/needZygote.jl")
	@require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" include("Utils/needFlux.jl")
end

end # module SimpleAlgebra
