#include <torch/extension.h>
#include <vector>
#include <ATen/Parallel.h>

template <typename scalar_t>
torch::Tensor ASL_Forward(const torch::Tensor &input, const torch::Tensor &theta)
/*
    In the forward pass we receive input tensor. 
	We must return a output tensor.
*/
{
	auto output = torch::zeros_like(input, input.options());

	int64_t N = input.size(0);
	int64_t C = input.size(1);
	int64_t H = input.size(2);
	int64_t W = input.size(3);

	int64_t inStrideN = input.stride(0);
	int64_t inStrideC = input.stride(1);
	int64_t inStrideH = input.stride(2);
	int64_t inStrideW = input.stride(3);

	scalar_t *inputPtr = input.data<scalar_t>();
	scalar_t *outputPtr = output.data<scalar_t>();
	scalar_t *thetaPtr = theta.data<scalar_t>();

	//loop through index
	at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
		for (int64_t n = start; n < end; ++n)
		{
			for (int64_t c = 0; c < C; ++c)
			{
				scalar_t *currInputPtr = inputPtr + n * inStrideN + c * inStrideC;
				scalar_t *currThetaPtr = thetaPtr + c * theta.stride(0);

				//alpha and beta define the  horizontal and verticalamount of shift. Eq (7)
				scalar_t alpha = currThetaPtr[0];
				scalar_t beta = currThetaPtr[theta.stride(1)];

				//Eq (9)
				int64_t alphaFloor = static_cast<int64_t>(std::floor(alpha));
				int64_t betaFloor = static_cast<int64_t>(std::floor(beta));
				int64_t alphaDiff = static_cast<int64_t>(alpha) - alphaFloor;
				int64_t betaDiff = static_cast<int64_t>(beta) - betaFloor;

				for (int64_t h = 0; h < H; ++h)
				{
					for (int64_t w = 0; w < W; ++w)
					{
						scalar_t *currOutputPtr = outputPtr + n * output.stride(0) + c * output.stride(1) + 
						h * output.stride(2) + w * output.stride(3);
						//Eq (10)
						scalar_t z1 = static_cast<scalar_t>(0);
						scalar_t z2 = static_cast<scalar_t>(0);
						scalar_t z3 = static_cast<scalar_t>(0);
						scalar_t z4 = static_cast<scalar_t>(0);
						if (h + alphaFloor < H && w + betaFloor < W)
						{
							z1 = currInputPtr[(h + alphaFloor) * inStrideH + (w + betaFloor) * inStrideW];
						}
						if (h + alphaFloor < H && w + betaFloor + 1 < W)
						{
							z2 = currInputPtr[(h + alphaFloor) * inStrideH + (w + betaFloor + 1) * inStrideW];
						}
						if (h + alphaFloor + 1 < H && w + betaFloor < W)
						{
							z2 = currInputPtr[(h + alphaFloor + 1) * inStrideH + (w + betaFloor) * inStrideW];
						}
						if (h + alphaFloor + 1 < H && w + betaFloor + 1 < W)
						{
							z2 = currInputPtr[(h + alphaFloor + 1) * inStrideH + (w + betaFloor + 1) * inStrideW];
						}
						//Eq(8)
						*currOutputPtr += z1 * (1 - alphaDiff) * (1 - betaDiff) + z2 * (alphaDiff) * (1 - betaDiff) + z3 * (1 - alphaDiff) * (betaDiff) + z4 * (alphaDiff) * (betaDiff);
					}
				}
			}
		}
	});
	return output;
}

template <typename scalar_t>
torch::Tensor ASL_Backward(const torch::Tensor &out_gradient, const torch::Tensor &input, const torch::Tensor &theta)
/*
    In the backward pass we receive out_gradient.  
	We must return in_gradient.
*/
{
	auto in_gradient = torch::zeros_like(out_gradient, out_gradient.options());

	int64_t N = input.size(0);
	int64_t C = input.size(1);
	int64_t H = input.size(2);
	int64_t W = input.size(3);

	int64_t inStrideN = input.stride(0);
	int64_t inStrideC = input.stride(1);
	int64_t inStrideH = input.stride(2);
	int64_t inStrideW = input.stride(3);

	int64_t inGStrideN = in_gradient.stride(0);
	int64_t inGStrideC = in_gradient.stride(1);
	int64_t inGStrideH = in_gradient.stride(2);
	int64_t inGStrideW = in_gradient.stride(3);

	scalar_t *inputPtr = input.data<scalar_t>();
	scalar_t *thetaPtr = theta.data<scalar_t>();
	scalar_t *out_gradientPtr = out_gradient.data<scalar_t>();
	scalar_t *in_gradientPtr = in_gradient.data<scalar_t>();

	// loop through index
	at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
		for (int64_t n = start; n < end; ++n)
		{
			for (int64_t c = 0; c < C; ++c)
			{
				scalar_t *currInputPtr = inputPtr + n * inStrideN + c * inStrideC;
				scalar_t *currThetaPtr = thetaPtr + c * theta.stride(0);

				scalar_t alpha = *currThetaPtr;
				scalar_t beta = currThetaPtr[theta.stride(1)];

				int64_t alphaFloor = static_cast<int64_t>(std::floor(alpha));
				int64_t betaFloor = static_cast<int64_t>(std::floor(beta));
				int64_t alphaDiff = static_cast<int64_t>(alpha) - alphaFloor;
				int64_t betaDiff = static_cast<int64_t>(beta) - betaFloor;

				for (int64_t h = 0; h < H; ++h)
				{
					for (int64_t w = 0; w < W; ++w)
					{
						scalar_t currOutGPtr = out_gradientPtr + n * out_gradient.stride(0) + c * out_gradient.stride(1) 
						+ h * out_gradient.stride(2) + w * out_gradient.stride(3);
						if (h + alphaFloor < H && w + betaFloor < W)
						{
							z1 = currInputPtr[(h + alphaFloor) * inStrideH + (w + betaFloor) * inStrideW];
						}
						if (h + alphaFloor < H && w + betaFloor + 1 < W)
						{
							z2 = currInputPtr[(h + alphaFloor) * inStrideH + (w + betaFloor + 1) * inStrideW];
						}
						if (h + alphaFloor + 1 < H && w + betaFloor < W)
						{
							z2 = currInputPtr[(h + alphaFloor + 1) * inStrideH + (w + betaFloor) * inStrideW];
						}
						if (h + alphaFloor + 1 < H && w + betaFloor + 1 < W)
						{
							z2 = currInputPtr[(h + alphaFloor + 1) * inStrideH + (w + betaFloor + 1) * inStrideW];
						}
						scalar_t *currInGPtr = in_gradientPtr + n * in_gradient.stride(0) + c * in_gradient.stride(1)
						 + h * in_gradient.stride(2) + w * in_gradient.stride(3);
						//tries to negate the change in gradient made in forward
						*currInGPtr -= z1 * (1 - alphaDiff) * (1 - betaDiff) + z2 * (alphaDiff) * (1 - betaDiff) + z3 * (1 - alphaDiff) * (betaDiff) + z4 * (alphaDiff) * (betaDiff);
						//Probably needs change ㅠㅠㅠㅠ
					}
				}
			}
		}
	});
	return in_gradient;
}

Tensor ASL_Forward_actual(const Tensor &input, const Tensor &theta)
{
	return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ASL_Forward_actual", [&] {
		return ASL_Forward<scalar_t>(input, theta);
	});
}

Tensor ASL_Backward_actual(const Tensor &out_gradient, const Tensor &input, const Tensor &theta)
{
	return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "ASL_Backward_actual", [&] {
		return ASL_Backward<scalar_t>(out_gradient, input, theta);
	});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &ASL_Forward_actual, "ASL_Forward");
	m.def("backward", &ASL_Backward_actual, "ASL_Backward");
}
