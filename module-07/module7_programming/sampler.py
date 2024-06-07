# External dependencies
import torch
import numpy as np

class Sampler:
	"""
	Class implementing a sampler for inference on a model. Given the raw logits from
	an LLM model, this will sample the next token ID.

	NOTE: If top_k and top_p are both None, sample from
	      the whole distribution (same as top_p=1.0).

	Arguments:
		top_k:             If specified, only the top k logits should be used
		                   during sampling. If this is specified, top_p should
						   be None.
		top_p:             If specified, only the logits representing the
		                   probability mass p should be used during sampling.
			               Or, if the top token has mass greater than p, the
						   top token is returned. If this is specified, top_k
						   should be None.
		frequency_penalty: A penalty applied to tokens that have previously
		                   occured in the sequence. Along with presence_penalty,
						   this adjusts the per-token softmax temperature.
			               A penalty of 1.0 indicates no change from normal softmax.
		presence_penalty:  A penalty applied to tokens IF they have previously
		                   occured in the sequence. Along with frequency_penalty,
						   this adjusts the per-token softmax temperature.
			               A penalty of 1.0 indicates no change from normal softmax.
	"""
	def __init__(
		self,
		top_k: int = None,
		top_p: int = None,
		frequency_penalty: float = 1.0,
		presence_penalty: float = 1.0
	) -> None:

		# TODO
		pass

	def sample_token(
		self, raw_unsorted_logits: np.ndarray[float],
		previous_token_ids: np.ndarray[int]
	) -> int:
		"""
		TBA

		Arguments:
			raw_unsorted_logits: A one dimensional list of logits representing
			                     an unnormalized distribution over next tokens
				                 These are "unsorted" in the sense that their
								 order aligns with vocabulary order, not with
								 probability.
			previous_token_ids:  A one dimensional list of ids representing the
			                     previous tokens, for calculating repetition
								 penalties.
		Return Values:
			sampled_token_id:    A single token ID, sampled according to the
			                     specified sampling parameters.
		"""

		# TODO

	# An alternative way to call sample_token(), for convenience
	def __call__(self, raw_unsorted_logits, previous_token_ids):
		return self.sample_token(raw_unsorted_logits, previous_token_ids)

if __name__ == '__main__':

	sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

	sequence = [1,2,3,4,5]

	for i in range(10):
		# fake logits for a vocab of size 500
		logits = np.random.randn(500)

		# get next token in sequence
		next_token = sampler(logits, sequence)
		sequence.append(next_token)

	print(sequence)
