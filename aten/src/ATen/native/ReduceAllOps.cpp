#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/Resize.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0,
              "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  min_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor& min_unary_out(const Tensor &self, Tensor& out) {
  Tensor tmp_output = at::min(self);
  at::native::resize_output(out, tmp_output.sizes());
  out.copy_(tmp_output);
  return out;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0,
              "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  max_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor& max_unary_out(const Tensor &self, Tensor& out) {
  // First check if the devices match (CPU vs GPU)
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));

  at::native::resize_output(out, {});

  max_all_stub(self.device().type(), out, self.contiguous());
  return out;
}

// DEPRECATED: Use at::aminmax instead
std::tuple<Tensor, Tensor> _aminmax_all(const Tensor &self) {
  TORCH_WARN_ONCE("_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
                  " This warning will only appear once per process.");
  return at::aminmax(self);
}

}} // namespace at::native
