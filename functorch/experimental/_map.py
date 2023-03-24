from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._ops import PyOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.utils._pytree import tree_flatten
from ._cond import _has_potential_branch_input_alias, _has_potential_branch_input_mutation, UnsupportedAliasMutationException


map = PyOperator("map")

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, args_spec, *flat_args):
        ctx.save_for_backward(*flat_args)
        ctx._f = f
        ctx._args_spec = args_spec
        xs, args = pytree.tree_unflatten(flat_args, args_spec)
        out = map(f, xs, *args)
        flat_out, out_spec =  pytree.tree_flatten(out)
        return out_spec, *flat_out
    
    @staticmethod
    def backward(ctx, grad_spec, *flat_grads):
        xs, args = pytree.tree_unflatten(ctx.saved_tensors, ctx._args_spec)
        def fw_bw(grad_out_and_x, *args):
            flat_grads, x = grad_out_and_x 
            with torch.enable_grad():
                # Since x and flat_grads are created in map_cpu, where autograd is disabled.
                # We create a view of args to allow autograd through them.
                x, args = pytree.tree_map(lambda t: t.view(t.shape), (x, args))
                out = ctx._f(x, *args)
            flat_out, _ = pytree.tree_flatten(out)
            def grad(arg):
                if arg.requires_grad:
                    return torch.autograd.grad(flat_out, arg, flat_grads, retain_graph=True)[0]
                return None
            return pytree.tree_map(grad, (x, *args))
        grads = map(fw_bw, (flat_grads, xs), *args)
        return None, None, *pytree.tree_flatten(grads)[0]

def trace_map(proxy_mode, func_overload, f, xs, *args):
    if not isinstance(xs, torch.Tensor):
        raise ValueError("map() must loop over a tensor")
    if len(xs.shape) == 0 or xs.shape[0] == 0:
        raise ValueError("map() cannot be traced with scalar tensors or zero dimension tensors")
    if not all(isinstance(o, torch.Tensor) for o in args):
        raise ValueError("map() operands must be a list of tensors or modules")

    with disable_proxy_modes_tracing():
        body_graph = make_fx(f)(xs[0], *args)

    next_name = None
    i = 0
    while not next_name:
        candidate = f"body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    proxy_mode.tracer.root.register_module(next_name, body_graph)
    node_args = (body_graph, xs, *args)
    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                               name="map")
    outs = [body_graph(x, *args) for x in xs]
    # Implementation notes: we need to use new_empty() + copy_() here instead of stack() directly
    # because stack([...]) takes a fixed size list which will specialize dynamic shape here.
    # Meanwhile we want to preserve the looped over dimension as symbolic shape, such that:
    # ys: Tensor[s0, ...] = map(xs: Tensor[s0, ...], *args)
    out = outs[0].new_empty([xs.shape[0], *outs[0].shape])
    out.copy_(torch.stack(outs))
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)

def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    assert all([isinstance(xs, torch.Tensor) for xs in flat_xs]), f"Leaves of xs must be Tensor {flat_xs}"
    assert all([xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs]), f"Leaves of xs must have same leading dimension size {flat_xs}"
    a = list(zip(*flat_xs))
    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees

def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    b = list(zip(*flat_out))
    stacked_out = []
    for leaves in b:
        if all([leave is not None for leave in leaves]):
            stacked_out.append(torch.stack(leaves))
        else:
            stacked_out.append(None)
    return pytree.tree_unflatten(stacked_out, out_spec)

@map.py_impl(DispatchKey.CUDA)
@map.py_impl(DispatchKey.CPU)
def map_cpu(f, xs, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU/CUDA key"
    return _stack_pytree(f(_unstack_pytree(xs), *args))


@map.py_impl(DispatchKey.Autograd)
def map_autograd(f, xs, *args):
    # We want to only disable Autograd for map because we need to autograd f in map.
    # This seems to prohibit autograd nested map.
    map.non_fallthrough_keys = map.non_fallthrough_keys.remove(DispatchKey.AutogradCPU)
    map.non_fallthrough_keys = map.non_fallthrough_keys.remove(DispatchKey.AutogradCUDA)
    # Autograd.Function can only handle flattend inputs and produce flattend outputs.
    flat_args, args_spec = pytree.tree_flatten((xs, args))
    out_spec, *flat_outs= MapAutogradOp.apply(f, args_spec, *flat_args)
    outs = pytree.tree_unflatten(flat_outs, out_spec)
    map.non_fallthrough_keys = map.non_fallthrough_keys.add(DispatchKey.AutogradCPU)
    map.non_fallthrough_keys = map.non_fallthrough_keys.add(DispatchKey.AutogradCUDA)
    return outs


@map.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, xs, *args):
    print("map forward proxy torch dispatch")
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_map(mode, map, f, xs, *args)
    return res


@map.py_impl(FakeTensorMode)
def map_fake_tensor_mode(f, xs, *args):
    print("map_fake_tensor")
    outs = [f(x, *args) for x in xs]
    return outs[0].new_empty([xs.shape[0], *outs[0].shape])

@map.py_impl(torch._C._functorch.TransformType.Functionalize)
def map_functionalize(interpreter, f, xs, *args):
    print("map_functionalize")
    """
    Functionalization implementation for torch.map. Currently:
      1. We don't allow any input mutation inside the map function
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(args, reapply_views=reapply_views)

    functional_map_fn = functionalize(f, remove=mode)

    with interpreter.lower():
        inputs = (unwrapped_xs,) + unwrapped_args
        if _has_potential_branch_input_mutation(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map(functional_map_fn, unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())

# TODO(voz) Make this automatic for keys, this is very ugly atm
map.fallthrough(DispatchKey.PythonDispatcher)
map.fallthrough(DispatchKey.PythonTLSSnapshot)
map.fallthrough(DispatchKey.ADInplaceOrView)
map.fallthrough(DispatchKey.BackendSelect)
map.fallthrough(DispatchKey.AutocastCPU)