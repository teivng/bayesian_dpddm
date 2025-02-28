# Change Log

Changed the computation of disagreement rates: 

- Instead of taking the argmax of the logits, we sample from a categorical distribution given by the logits instead.

In this way, the temperature parameter actually influences the stochasticity of the sampled predictions.

