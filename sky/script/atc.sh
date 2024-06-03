set -x
bs=1
img_size=224
atc --framework=5 --model=$1 \
	--output=om/out --input_format=NCHW \
	--input_shape="input:${bs},3,${img_size},${img_size}" \
	--log=debug --soc_version=Ascend310 \
	--enable_small_channel=1 --optypelist_for_implmode="Gelu" \
	--op_select_implmode=high_performance
