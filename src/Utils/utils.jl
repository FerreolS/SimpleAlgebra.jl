#= function infer_size(f::Function,args...)
	size(Broadcast.broadcasted(Broadcast.combine_styles(args...),  f,args...))
end

infer_size(args...) = infer_size(+, args...)
 =#

narrow_type(T)  = T
narrow_type(T::Tuple)  = narrow_type(T...)
narrow_type(::Type{T}, ::Type{T}) where {T} = T
narrow_type(::Type{T1},::Type{T2}) where {T1,T2<:T1} = T2
narrow_type(::Type{T2},::Type{T1}) where {T1,T2<:T1} = T2
narrow_type(T, S, U, V...) = (@inline; narrow_type(T, narrow_type(S, U, V...)))
narrow_type(::Type{T1},::Type{T2}) where {T1,T2} = throw(SimpleAlgebraFailure("Type $T1 and $T2 cannot be narrowed to a common type"))

function extend_axes(ax::NTuple{N,T},len) where {N,T}
    @inline
   len::Integer <= N ? ax[1:len] : (ax..., (Base.OneTo(1) for i ∈ (N+1):len)...)
end