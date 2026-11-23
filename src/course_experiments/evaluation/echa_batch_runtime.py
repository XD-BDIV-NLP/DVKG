"""单批核验期间复用不可变输入摘要，批前批后仍校验完整输入。"""
from functools import wraps
from types import FunctionType,SimpleNamespace

def clone(function,overrides):
    return FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__,function.__defaults__,function.__closure__)

def cached_batch_verifier(original):
    @wraps(original)
    def verify(provider,graph,structure,plans,root,stage):
        digest=original.__globals__['digest'];core=original.__globals__['core']
        identities={id(value):(value,digest(value)) for value in (graph,structure)}
        hierarchy=[]
        def memo(value):
            saved=identities.get(id(value))
            return saved[1] if saved is not None and value is saved[0] else digest(value)
        def tree(value):
            if value is not structure:return core._hierarchy(value)
            if not hierarchy:hierarchy.append(core._hierarchy(value))
            return hierarchy[0]
        proxy=SimpleNamespace(**vars(core))
        proxy.lift_coverage=clone(core.lift_coverage,dict(digest=memo,_hierarchy=tree))
        function=clone(original,dict(digest=memo,core=proxy))
        try:return function(provider,graph,structure,plans,root,stage)
        finally:
            if any(digest(value)!=expected for value,expected in identities.values()):
                raise ValueError('Frozen graph or textbook changed during batch verification')
    return verify
