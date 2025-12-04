import os
import orbax.checkpoint as ocp
from typing import Any, Optional

class Checkpoint:
    def __init__(self, directory: str, max_to_keep: int = 2):
        self.directory = os.path.abspath(directory)
        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
        # Use PyTreeCheckpointer for standard JAX PyTrees (params)
        self.manager = ocp.CheckpointManager(
            self.directory, 
            ocp.PyTreeCheckpointer(), 
            options
        )

    def save(self, step: int, params: Any, metadata: Optional[dict] = None):
        """Save parameters and optional metadata."""
        # For PyTreeCheckpointer, we can just pass the pytree.
        # Metadata support depends on the item being saved or separate handling.
        # Orbax CheckpointManager saves 'items'.
        # If we want to save metadata, we can wrap it or save separately.
        # For simplicity, we'll assume params is the main thing.
        # If metadata is needed, we can bundle it: {'params': params, 'metadata': metadata}
        
        save_item = params
        if metadata is not None:
            save_item = {'params': params, 'metadata': metadata}
            
        self.manager.save(step, save_item)
        # Wait for save to complete to ensure consistency if needed, 
        # but usually async is fine. 
        self.manager.wait_until_finished()

    def load(self, step: int = None):
        """Load parameters. Returns the whole saved item."""
        if step is None:
            step = self.manager.latest_step()
            if step is None:
                return None
        
        return self.manager.restore(step)

    def latest_step(self):
        return self.manager.latest_step()
