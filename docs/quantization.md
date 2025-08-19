# Quantization

Quantization is refers to running models at a lower precision datatype (eg. fp8, int8) than they were trained on. A good overview of quantization can be found in the huggingface documentation [here](https://huggingface.co/docs/transformers/en/main_classes/quantization). We provide a brief overview below.

## Quantization Overview

When quantizing models, particularly from float to integer datatypes, we remap the real-valued number range to a range of integers. This mapping is represented by some real valued numbers/parameters (For example, in *affine quantization* we have a scaling factor and a zero point). When we map to a lower precision data type, we lose, well precision, often leading to some predictive performance loss. However we also reduce the memory overhead during inference, allowing us to make more predictions, faster with less computational resources. This make quantization a trade-off between predictive performance and computational efficiency.

When we map from a higher precision datatype to a lower precision datatype, we do not necessarily need to map the entire range of representable values. If the set of values we want to quantize sit within a smaller range, then we can use this smaller range when mapping to the lower precision datatype, which results in less precision loss (less range + same number of steps = smaller steps). There are a multitude of techniques for minimizing not only the loss in precision, but also losses in predictive performance when quantizing LLM's, however we have just described the basics at a simple level.

## Quantization Block Format

Quantization block format refers to the granularity at which you quantize a set of values. For example, we could quantize the entire set of model weights globally, applying a single mapping from the high precision datatype to the low precision datatype. With simple affine quantization this would mean that we have a single scaling factor and zero point value for the entire model. This is almost never done and often higher granularity is used.

### Layer-wise Quantization

Each layer is quantized independently, recieving its own unique mapping from the high precision data type to the low precision data type.

### Channel-wise Quantization

Even more granularity than layer-wise quantization. Each individual channel within each layer is mapped independetly to the lower precision data type.

### Group-wise Quantization

This is a generic term to represent other levels of granularity, often between channel-wise and layer-wise. For example, one could quantize groups of channels together.

## Quantization Targets

Many libraries support quantizing only parts of the model inference (referred to as mixed precision). This may involve casting types back and forth in some cases, but often involves special compute kernels for performing operations between different data types. At a high level (there may be advanced implementations that don't follow this assumption) the three parts (or tensors) within the inference pipeline that one can quantize are the model weights, the activations and the KV cache. KV cache quantization is kind of its own thing and is independent from the other two.

| Quantization Type | Weights | Activations | KV Cache | Desc |
|---------|-------------|----------|------|------|
| Full Quantization | quantized | quantized | either | Best throughput and efficiency gains but greatest risk of performance loss. Computations actually occur at lower precision, which is great if you've got optimized low precision compute kernels on the gpu or serving engine |
| Weights Only Quantization | quantized | full precision | full precision | More balanced that full quantization, memory footprint is still reduced, but less efficiency and throughput gains since computations are still done at full precision |
| Activation Only Quantization | full precision | quantized | either | Not a thing, memory footprint is not reduced and little to no throughput/efficiency gains since computations still need to happen at full precision.

### Weight Quantization
Quantizing the weights is generally pretty easy since we know the minimum and maximum weights at quantization time. However there are several more sophisticated approaches. For example some approaches try to remove outliers, others use a calibration data set and quantize the weights one at a time, slightly modifying the mapping each time to minimize error. Quantizing the weights helps reduce the memory footprint of the model, allowing us to fit the model on less gpu's.

Update: So in LLM's, basically all of the parameters/compute are linear layers. Trying to quantize other layers usually hurts performance disproportionally to the memory savings. So most frameworks only quantize the linear layers.

### Activation Quantization
Quantizing the activations helps reduce the computational efficiency of model inference. It's done mostly as an extension to weight quantization and is never really done on its own. The benefit is that when both the weights and activations are quantized to the same data type, the compute operations can actually occur at low precision rather than in mixed precision (like with weight-only quantization). This reduces the computational overhead for compute bound applications since if our gpu supports the low precision datatype, then it can likely utilize its hardware to perform more operations per second with that datatype. However LLM deployments are often bound by memory bandwidth. Activation quantization helps throughput by further reducing their memory footprint, which means we can read and write activations to memory faster and take up less of the memory bandwidth.

### KV Cache Quantization

KV cache quantization is kind of its own thing and can be used separately from weight-only or weight and activation quantization. Quantizing the KV cache helps reduce the memory footprint, especially for really long context windows. Similar to activation quantization it can also increase throughoutput in scenarios where memory bandwidth is the limiting factor (which is often the case for LLM deployment) since the KV cache needs to be repeatedly accessed.

## Activation Quantization Approaches

### Training Aware Quantization

This involves using quantization during training to compute the quantization error while the model is being trained so that it can optimize itself in such a way that will minimize the loss in predictive performance when it is quantized after training.

### Post Training Quantization

Post training quantization (PTQ) refers to quantizing a model after it has already been trained.

#### Dynamic PTQ

In dynamic PTQ, the values are quantized on the fly at runtime. This adds a lot of computational overhead but results in minimal performance loss. This is usually only used for the activations, so you can generally assume that dynamic-PTQ always refers to quantization of the activations. Dynamic activation quantization needs to be supported by the model runtime framework that you are using.

#### Static PTQ

Static PTQ refers to having a predefined set of quantization mappings prior to inference runtime. Static PTQ is pretty much always the approach used for weight quantization, but can also be used for activations as well in some instances. These predefined mappings are often computed by using a calibration dataset to monitor the activations and estimate optimal mappings. The downside to this approach for activation quantization is that it is possible that during runtime you will have activations that fall outside the range seen during the calibration phase, resulting in them having to be clipped and potential performance loss. Thats kind of a simplification but basically it works less well than dynamic PTQ but runs much faster.

## Quantization Algorithms

In summary, the most common types of quantization are weight-only static PTQ quantization or static PTQ weight quantization with dynamic PTQ activation quantization.

<h3 style="display:inline">GPTQ</h3> <em>static-PTQ, weight-only, group-wise</em>

**G**enerative **P**retrained **T**ransformers **Q**uantization (GPTQ) is an algorithm for quantizing the weights of LLM's. It processes one layer at a time, quantizing a subset of the weights within the layer and using a small calibration set to observe the resultant error. It then adjusts the unquantized weights to compensate for the quantization error and repeats the process. It was the standard weight-quantization approach for awhile.

<h3 style="display:inline">AWQ</h3> <em>static-PTQ, weight-only, channel-wise, mixed-precision</em>

**A**ctivation-Aware **W**eight **Q**uantization (AWQ) is a weight quantization algorithm that only partially quantizes model weights. It works by using a calibration dataset to identify important weights which it leaves at full precision, and then quantizes the remaining weights in a channel-wise fashion. This results in a mixed-precision model since some weights are at the original precision and some have been quantized to a lower precision. It results in less loss of prediction performance than GPTQ and for reasons I don't fully understand inference on AWQ quantized models is also faster than GPTQ quantized models. This makes AWQ better than GPTQ in basically every way.

Update: So apparently this may only be really used when targeting w4a16 (int4 weights bfp16 activations). Not sure why its not used for w8a8. Documentation surrounding these algorithms and compression schemes is really bad.

Also AWQ works best with asymettric quantization since the activations are not usually quantized, only the weights.

<h3 style="display:inline">SmoothQuant</h3> <em>activation-quantization</em>

SmoothQuant doens't actually quantize anything. Instead what it does is *smooth* the activations to reduce their range so that they are easier to quantize with higher precision. It does this by scaling the weights of the model based on the activations it sees during calibration with a calibration dataset to remove outlier activations. It's important to apply SmoothQuant smoothing before weight quantization. It originally was meant to be used with static PTQ activation quantization but many libraries apply it even when doing dynamic PTQ activation quantization in order to reduce the activation range within batches.

## Quantization Libraries

In this repository we are going to focus on quantization libraries that are compatible with [vLLM](https://docs.vllm.ai/en/latest/features/quantization/index.html).

It is important to note that vLLM does not support static activation quantization

For now I'm only going to document llmcompressor since that what we're using at the moment. Want to get started on testing/implementation

### llmcompressor

### llama.cpp

### compressed-tensors

