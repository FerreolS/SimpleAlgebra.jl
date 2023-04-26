module SimpleAlgebra
using ChainRulesCore

struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Domain.jl")
include("Map.jl")
include("Cost/Cost.jl")
include("LinOp/LinOp.jl")
end # module SimpleAlgebra
