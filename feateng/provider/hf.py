from typing import List

import torch
from evalplus.provider.hf import HuggingFaceDecoder as OrgHuggingFaceDecoder

from feateng.provide.utility import make_raw_chat_prompt


class HuggingFaceDecoder(OrgHuggingFaceDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_new_tokens = 4096

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = make_raw_chat_prompt(prompt, self.tokenizer)
        self.force_base_prompt = True  # Hack. We use chat template anyway
        self.eos = ["\n```\n"]
        return OrgHuggingFaceDecoder.codegen(self, prompt, do_sample, num_samples)
