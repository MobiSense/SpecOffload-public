from typing import List
from threading import Event as ThreadEvent
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event as ProcessEvent
import torch

class MultiThreadGenerationInterface:
    def __init__(
        self,
        batch_finish_events: List[ThreadEvent],
        alive_flags: List[bool],
        thread_id: int = 0,
    ):
        self.thread_id = thread_id
        self.batch_finish_events = batch_finish_events
        self.alive_flags = alive_flags
        
    def wait_and_clear(self):
        if not self.alive_flags[1 - self.thread_id]:
            return
    
        wait_event = self.batch_finish_events[1 - self.thread_id]
        wait_event.wait()
        wait_event.clear()
        
    def set(self):
        set_event = self.batch_finish_events[self.thread_id]
        set_event.set()
        
    def set_finished(self):
        self.alive_flags[self.thread_id] = False
        
class AssistantProcessInterface:
    def __init__(
        self,
        vocab_size: int,
        input_token_length: int,
        max_new_tokens: int,
        generation_batch_size: int,
        assistant_batch_size: int,
        assistant_max_new_tokens: int = 5,
        number_of_threads: int = 1,
    ):
        # Parameters
        self.generation_batch_size = generation_batch_size
        self.assistant_batch_size = assistant_batch_size
        self.max_new_tokens = max_new_tokens
        self.vocab_size = vocab_size
        self.assistant_max_new_tokens = assistant_max_new_tokens
        
        # Shared memory
        self.token_length = mp.Value('i')
        self.new_token_length = mp.Value('i')
        self.input_token_length = mp.Value('i')
        self.input_ids = torch.LongTensor(size=(
            2 * generation_batch_size *
            (input_token_length + max_new_tokens * (assistant_max_new_tokens + 2))
        ,))
        self.attention_mask = torch.LongTensor(size=(
            2 * generation_batch_size *
            (input_token_length + max_new_tokens * (assistant_max_new_tokens + 2))
        ,))
        self.candidate_logits = torch.FloatTensor(size=(
            2 * generation_batch_size *
            assistant_max_new_tokens *
            vocab_size
        ,))
        
        # Synchorinzation events
        self.assistant_ready_event = mp.Event()
        self.input_ready_event = mp.Event()
        self.output_ready_event = mp.Event()
        
        self.batch_finish_prefilling_events = [mp.Event() for _ in range(number_of_threads)]
        
    def get_input_ids(self) -> torch.LongTensor:
        batch_size = self.generation_batch_size
        token_length = self.token_length.value
        
        return self.input_ids[:batch_size * token_length].reshape(
            batch_size, token_length
        ).clone()
        
    def set_input_ids(self, input_ids: torch.LongTensor) -> None:
        self.token_length.value = input_ids.shape[1]
        self.input_ids[:input_ids.numel()] = input_ids.reshape((input_ids.numel(),))

    def get_attention_mask(self) -> torch.LongTensor:
        batch_size = self.generation_batch_size
        token_length = self.token_length.value
        
        return self.attention_mask[:batch_size * token_length].reshape(
            batch_size, token_length
        ).clone()
        
    def set_attention_mask(self, attention_mask: torch.LongTensor) -> None:
        self.attention_mask[:attention_mask.numel()] = attention_mask.reshape((attention_mask.numel(),))

    def get_candidate_logits(self) -> torch.FloatTensor:
        batch_size = self.generation_batch_size
        new_token_length = self.new_token_length.value
        vocab_size = self.vocab_size
        
        return self.candidate_logits[:batch_size * new_token_length * vocab_size].reshape(
            batch_size, new_token_length, vocab_size
        ).clone()
        
    def set_candidate_logits(self, candidate_logits: torch.FloatTensor) -> None:
        self.new_token_length.value = candidate_logits.shape[1]
        self.candidate_logits[:candidate_logits.numel()] = candidate_logits.reshape((candidate_logits.numel(),))