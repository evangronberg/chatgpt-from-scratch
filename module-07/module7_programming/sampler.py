# External dependencies
import torch
import numpy as np

class Sampler:
	"""
	Class implementing a sampler for inference on a model. Given the raw logits from
	an LLM model, this will sample the next token ID.

	NOTE: If top_k and top_p are both None, sample from
	      the whole distribution (same as top_p = 1.0).

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
		top_p: float = None,
		frequency_penalty: float = 1.0,
		presence_penalty: float = 1.0
	) -> None:

		if top_k is not None and top_p is not None:
			raise ValueError('top_k and top_p may not BOTH be specified!')

		self.top_k = top_k
		self.top_p = top_p
		if top_k is None and top_p is None:
			self.top_p = 1.0

		self.frequency_penalty = frequency_penalty
		self.presence_penalty = presence_penalty

	def sample_token(
		self, raw_unsorted_logits: np.ndarray[float],
		previous_token_ids: np.ndarray[int]
	) -> int:
		"""
		Selects the next token using either top-k or top-p sampling,
		with frequency and presence penalties applied to previously
		occuring words.

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
		logits = torch.tensor(raw_unsorted_logits)
		logits, indices = torch.sort(logits, descending=True)

		penalties = torch.ones((len(logits)))
		occurred_token_ids = set()
		for token_id in previous_token_ids:
			penalties[token_id] += (self.frequency_penalty - 1)
			if token_id not in occurred_token_ids:
				penalties[token_id] += (self.presence_penalty - 1)
				occurred_token_ids.add(token_id)

		if self.top_p is not None:
			logits = torch.softmax(logits / penalties, dim=0)
			p_sum, top_p_tokens = 0.0, []
			for logit, index in zip(logits, indices):
				if p_sum < self.top_p:
					p_sum += float(logit)
					top_p_tokens.append(int(index))
			top_p_distribution = torch.softmax(
				logits[top_p_tokens] / penalties[top_p_tokens], dim=0)
			top_p_distribution_sample_index = torch.multinomial(
				top_p_distribution, num_samples=1)
			sampled_token_id = top_p_tokens[top_p_distribution_sample_index]
		else:
			top_k_tokens = indices[:self.top_k]
			top_k_distribution = torch.softmax(
				logits[:self.top_k] / penalties[:self.top_k], dim=0)
			top_k_distribution_sample_index = torch.multinomial(
				top_k_distribution, num_samples=1)
			sampled_token_id = top_k_tokens[top_k_distribution_sample_index]

		return sampled_token_id

	# An alternative way to call sample_token(), for convenience
	def __call__(
		self, raw_unsorted_logits: np.ndarray[float],
		previous_token_ids: np.ndarray[int]
	) -> int:
		return self.sample_token(raw_unsorted_logits, previous_token_ids)

if __name__ == '__main__':

	sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

	sequence = [1,2,3,4,5]

	for i in range(10):
		# Fake logits for a vocab of size 500
		logits = np.random.randn(500)

		# Get next token in sequence
		next_token = sampler(logits, sequence)
		sequence.append(next_token)

	print(sequence)
