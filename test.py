import torch
from torch._higher_order_ops.scan import _fake_scan, scan

class ChunkedCE(torch.nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, scan_op, _input, weight, target, bias):
        CHUNK_SIZE = self.chunk_size

        def compute_loss(input_chunk, weight, bias, target):
            logits = torch.addmm(bias, input_chunk, weight.t())
            logits = logits.float()
            loss = self.ce(logits, target)
            return loss

        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        loss_acc = torch.zeros((), device=_input.device)

        chunks = _input.shape[0] // CHUNK_SIZE

        _input_chunks = _input.view(chunks, CHUNK_SIZE, *_input.shape[1:])
        target_chunks = target.view(chunks, CHUNK_SIZE, *target.shape[1:])

        def combine_fn(carry, xs):
            grad_weight, grad_bias, loss_acc = carry
            input_chunk, target_chunk = xs
            (
                chunk_grad_input,
                chunk_grad_weight,
                chunk_grad_bias,
            ), chunk_loss = torch.func.grad_and_value(
                compute_loss, argnums=(0, 1, 2)
            )(
                input_chunk, weight, bias, target_chunk
            )
            return (
                (
                    grad_weight + chunk_grad_weight,
                    grad_bias + chunk_grad_bias,
                    loss_acc + chunk_loss,
                ),
                chunk_grad_input,
            )

        (grad_weight, grad_bias, loss_acc), grad_inputs = scan_op(
            combine_fn,
            (grad_weight, grad_bias, loss_acc),
            (_input_chunks, target_chunks),
        )
        return (
            grad_weight / chunks,
            grad_bias / chunks,
            loss_acc / chunks,
            grad_inputs.view(-1, *_input.shape[1:]) / chunks,
        )

# chunk size 1024
mod = ChunkedCE(1024)
# Varying B or T to see the effect of the chunking.
# B, T, D, V = 16, 1024, 768, 128256
B, T, D, V = 32, 1024, 768, 128256
torch.set_default_device('cuda')
model = torch.nn.Linear(D, V).to(torch.bfloat16)
x = torch.randn(B, T, D, requires_grad=False, dtype=torch.bfloat16)
label = torch.randint(0, V, (B, T)).to(torch.int64)

inp, weight, target, bias = (
        x.view(-1, D).contiguous(), torch.rand_like(model.weight, requires_grad=False), label.view(-1).contiguous(), torch.randn_like(model.bias, requires_grad=False)
    )

def compiled_run():
    return torch.compile(mod, fullgraph=True)(scan, inp, weight, target, bias)

def loop_run():
    return torch.compile(mod, fullgraph=True)(_fake_scan, inp, weight, target, bias)

torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.optimize_scatter_upon_const_tensor = False
compiled_run()

# run memory profiling
def profile_mem(fn, test_name: str):
    torch._dynamo.reset()
    for _ in range(2):
        fn()
    torch.cuda.memory._record_memory_history()
    fn()
    torch.cuda.memory._dump_snapshot(f"{test_name}.pickle")

def bench_time(fn, test_name:str):
    torch._dynamo.reset()
    import time
    start = time.time()
    fn()
    end = time.time()
    print(f"{test_name} scan end - start", end - start)


# profile_mem(compiled_run, "scan")
# profile_mem(loop_run, "loop")
