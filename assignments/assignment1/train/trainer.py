import torch
import torch.nn as nn
from typing import Any, Optional, Union
from pathlib import Path
import time
import os


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        max_steps: int = 1000,
        global_batch_size: int = 32,
        micro_batch_size: int = 8,
        log_every_n_steps: int = 10,
        save_every_n_steps: Optional[int] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_steps = max_steps
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Calculate gradient accumulation steps
        assert global_batch_size % micro_batch_size == 0, \
            f"Global batch size ({global_batch_size}) must be divisible by micro batch size ({micro_batch_size})"
        self.grad_accumulation_steps = global_batch_size // micro_batch_size
        
        # Training state
        self.current_step = 0
        self.loss_accumulator = 0.0
        self.batch_count = 0
        
        # Setup checkpointing
        if save_every_n_steps is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
        
        # Timing
        self.start_time = None
    
    def training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Single training step with forward and backward pass."""
        # Forward pass
        logits = self.model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        
        # Backward pass (scaled for gradient accumulation)
        (loss / self.grad_accumulation_steps).backward()
        
        return loss

    def train(self, dataloader, profiler: Optional[Any] = None) -> None:
        """Main training loop."""
        self.model.train()
        self.start_time = time.time()
        
        print(f"Starting training for {self.max_steps} steps")
        
        for batch in dataloader:
            if self.current_step >= self.max_steps:
                break
            
            # Unpack batch
            inputs, targets = batch
            
            # Execute training step (forward + backward only)
            loss = self.training_step(inputs, targets)
            self.loss_accumulator += loss.item()
            
            # Check if this is an optimizer step
            self.batch_count += 1
            should_step = self.batch_count % self.grad_accumulation_steps == 0
            
            # Optimizer step after gradient accumulation
            if should_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                # Logging
                if self.current_step % self.log_every_n_steps == 0:
                    avg_loss = self.loss_accumulator / self.grad_accumulation_steps
                    lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - self.start_time
                    
                    print(f"step={self.current_step} | loss={avg_loss:.4f} | lr={lr:.2e} | time={elapsed:.1f}s")
                
                # Checkpointing
                if (self.save_every_n_steps is not None and 
                    self.current_step > 0 and 
                    self.current_step % self.save_every_n_steps == 0):
                    checkpoint_path = f"{self.checkpoint_dir}/checkpoint_step_{self.current_step}.pt"
                    self.save_checkpoint(checkpoint_path)
                    print(f"Saved: {checkpoint_path}")
                
                self.loss_accumulator = 0.0
                self.current_step += 1
            
            # Step profiler for schedule to work correctly
            if profiler is not None:
                profiler.step()
        
        print(f"Training completed in {time.time() - self.start_time:.1f}s")
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.current_step,
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint.get('step', 0)
        
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        print(f"Loaded checkpoint from step {self.current_step}")
