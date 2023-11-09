allsubtypes(T) = (s = subtypes(T); union(s, (allsubtypes(K) for K in s)...))

function Functors.functor(::Type{<:LinOpScale{I,T}},x) where {I,T}

    function reconstruct_Map(scale)
        return LinOpScale(size(I),scale)
    end
    y = x.scale
    return y, reconstruct_Map
end

for T in filter(!isabstracttype, allsubtypes(AbstractMap))
    @eval @functor T
end