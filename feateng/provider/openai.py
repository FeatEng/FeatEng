from typing import List

from evalplus.gen.util import openai_request
from evalplus.provider.openai import OpenAIChatDecoder as OrgOpenAIChatDecoder


class OpenAIChatDecoder(OrgOpenAIChatDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # O1 requires more tokens for 'thinking'
        self.max_new_tokens = 4096 if "o1" not in self.name else 16000

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)
        ret = openai_request.make_auto_request(
            self.client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
        )

        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)

        return outputs
