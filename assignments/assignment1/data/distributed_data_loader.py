import os
import torch
from pathlib import Path
from typing import Iterator, Tuple, List, Union

from data.data_loader import KJJ0DataLoader


class DistributedKJJ0DataLoader(KJJ0DataLoader):
    """
    Distributed version of KJJ0DataLoader that splits data across ranks.
    
    Key concept: All ranks read the same files in the same order, but each rank
    processes different contiguous chunks of tokens.
    
    Example with local_batch_size=2, sequence_length=4, world_size=2:
        - Rank 0 reads tokens[0:9]   for batch (8 tokens + 1 for target)
        - Rank 1 reads tokens[8:17]  for batch (8 tokens + 1 for target)
        - Both ranks then advance by 2 * 8 = 16 tokens
    
    This ensures:
    1. Each rank sees different data
    2. All ranks process data from the same global sequence
    3. Training is deterministic and equivalent to single-GPU training
    
    Args:
        file_paths: List of .bin file paths
        local_batch_size: Number of sequences per batch (per rank)
        sequence_length: Length of each sequence
        device: Device to place tensors on
        rank: Process rank (auto-detected from RANK env var if None)
        world_size: Total number of processes (auto-detected from WORLD_SIZE env var if None)
    """
    
    def __init__(
        self, 
        file_paths: List[Union[str, Path]], 
        local_batch_size: int, 
        sequence_length: int, 
        device: str = 'cpu',
        rank: int = None,
        world_size: int = None
    ):
        # TODO 1: Auto-detect distributed environment from environment variables
        # Hint: Use os.environ.get('RANK', 0) and os.environ.get('WORLD_SIZE', 1)
        # These variables are set automatically by torchrun
        self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))  # rank if rank is not None else int(os.environ.get('RANK', 0))
        self.world_size = world_size if world_size is not None else int(os.environ.get('WORLD_SIZE', 1))  # world_size if world_size is not None else int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize parent class
        super().__init__(file_paths, local_batch_size, sequence_length, device)
        
        self.local_batch_size = local_batch_size
       
        print(f"Distributed dataloader: rank={self.rank}, world_size={self.world_size}")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batches with rank-based partitioning.
        
        Yields:
            (input_batch, target_batch) tensors of shape [local_batch_size, sequence_length]
        """
        # Reset state for new iteration
        self.current_shard_idx = 0
        self.current_tokens = None
        self.current_position = 0
        
        # TODO 2: Number of tokens needed per rank per batch
        num_tokens_local = self.local_batch_size * self.sequence_length  # self.local_batch_size * self.sequence_length
        
        while True:
            try:
                # Load shard if needed
                while self.current_tokens is None or self.current_position + self.world_size * num_tokens_local >= len(self.current_tokens):
                    if self.current_shard_idx >= len(self.files):
                        return  # No more data
                    
                    self.current_tokens = self._load_shard(self.files[self.current_shard_idx])
                    self.current_shard_idx += 1
                    self.current_position = 0
                
                # TODO 3: Calculate this rank's starting position in the token stream
                # Hint: current_position is the global position, add an offset based on rank
                # Each rank needs num_tokens_local, so rank 0 starts at position 0,
                # rank 1 starts at position num_tokens_local, etc.
                pos_local = self.current_position + self.rank * num_tokens_local  # self.current_position + self.rank * num_tokens_local
                
                # Extract tokens (need +1 extra token for target shift)
                buf = self.current_tokens[pos_local : pos_local + num_tokens_local + 1]
                
                if len(buf) < num_tokens_local + 1:
                    # Not enough tokens remaining in shard, move to next
                    continue
                
                # Reshape into batch
                input_batch = buf[:-1].view(self.local_batch_size, self.sequence_length)
                target_batch = buf[1:].view(self.local_batch_size, self.sequence_length)
                
                # TODO 4: Advance the global position for the next iteration
                # Hint: All ranks need to advance by the total tokens consumed across all ranks
                # If we have 2 ranks each consuming 8 tokens, advance by 2 * 8 = 16
                self.current_position += self.world_size * num_tokens_local  # self.world_size * num_tokens_local
                
                yield input_batch.long().to(self.device), target_batch.long().to(self.device)
                
            except Exception as e:
                # Handle any unexpected errors
                print(f"Rank {self.rank} encountered error: {e}")
                break
