from typing import List

from evalplus.gen.util import anthropic_request
from evalplus.provider.anthropic import AnthropicDecoder as OrgAnthropicDecoder


class AnthropicDecoder(OrgAnthropicDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_new_tokens = 4096

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Greedy only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            message = anthropic_request.make_auto_request(
                client=self.client,
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop_sequences=self.eos + ["\n```\n"],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            outputs.append(message.content[0].text)

        return outputs

    def is_direct_completion(self) -> bool:
        return False
