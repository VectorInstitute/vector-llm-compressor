# Vector LLM Compressor Docs

This is just going to be a place for some generic documentation surrounding LLM quantization.

As a short summary:
- documentation on quantization algorithims, different quantization schemes, and what should be used for what is very poor
- As best I can tell, ampere series GPU's have very good int8 cores, but not as good int4 cores. So targeting w8a8 is probably best
- For some reason AWQ seems to be specifically for w4a16.
- Some indication that w4a16 is better for memory bound applications and w8a8 is better for compute bound applications. Logic is that w8a8 reduces compute by using