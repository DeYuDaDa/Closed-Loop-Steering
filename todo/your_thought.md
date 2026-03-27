The user wants to implement Strategy B: batched forward with KV cache padding across all active slots, so we get true GPU parallelism (one batched forward per step instead of N serial forwards).

Let me think through the architecture:

Strategy B Core Design:

Each decode step:

Collect last_token from all active slots → [N, 1]

Find max KV length across all slots

Left-pad all KV caches to max length

Build attention mask: for each slot, 1s for real tokens, 0s for padded positions, then 1 for the new token

Do ONE batched model() forward → [N, 1, vocab_size]

Extract per-slot logits and new KV values, then update each slot's KV cache by splitting out the new position from the stacked return

Run StateMonitor per slot (still serial but no GPU work) and sample per slot before checking done conditions

Key challenges:

Padding KV caches
: Each KV cache is a tuple of (key, value) per layer with shapes [1, num_heads, seq_len, head_dim], so I need to left-pad the seq_len dimension.

Attention mask
: For each slot, I need 1s for real tokens, 0s for padded positions, then 1 for the new token position.

Extracting updated KV caches
: After the batched forward, the returned past_key_values contains all positions up to max_kv_len + 1, and I need to track that each slot now has that extended length even though they're all padded to the same size.

Actually, the simpler approach is to just keep the returned past_key_values as-is without re-padding. Since all KV caches were padded to max_kv_len before the forward pass, they're now all length max_kv_len + 1, and the garbage values at padding positions will naturally get masked out by the attention mask in subsequent steps. When I do the next forward pass, I'll find the new max KV length, pad everything again, and the attention mask will handle masking out the old padding positions for shorter slots. gets complicated. Let me think differently.

Simpler approach
: Align all slots to the same KV length by left-padding EACH TIME. The key insight is:

After the first batched forward with max_kv_len, ALL slots' KV caches are now of length max_kv_len+1 (HF returns the full KV cache including the new position). For slots that were shorter, their KV cache was padded, and the forward produced "garbage" attention for the padded positions (but masked by attention mask). The returned KV for those slots at the padded positions is also "garbage", but it will always be masked in subsequent forwards.

So we can just keep each slot's past_key_values as the full [batch_slice] from the returned stacked KV, and always maintain per-slot attention masks that correctly mask out the padded positions.

After the batched forward, I need to split the returned KV caches back to individual slot caches—the returned shape is [N, heads, max_len+1, head_dim] and I need to separate this into N individual caches of [1, heads, max_len+1, head_dim] for each slot. Then in the next step, the shorter slots now have padded KV caches where their real content only starts partway through, so I need to track both the actual content length and the physical cache length for each slot to properly mask attention. After each generation round, all slots get padded to the same physical length, and each slot tracks how many real tokens it actually contains—the input length plus however many tokens we've generated so far. The attention mask then just marks which positions are real content versus padding, with 0s for the padded prefix and 1s for the actual tokens.

For the next forward pass, I'm stacking the KV caches from all slots into a batch, adding the new generated token to the sequence length, and running attention across all of them at once. Once the model returns the updated KV cache, I split it back out by slot so each one has its own cache for the next iteration.

The overall algorithm starts with a prefill phase where each slot gets processed independently to cache its prompt, then transitions into a generation phase where all slots run together in a batch.

But I'm realizing the initialization needs adjustment—I should sample the first token during prefill but hold off on appending it to the input IDs until I'm ready to start the batched decode loop, so the KV cache length stays synchronized with what I'm actually tracking. I'm extracting the last token from the input sequence and feeding it to the model along with the cached KV states from the prompt, making sure the attention mask dimensions align correctly—the mask should span the prompt length plus the new tokens being processed. After this forward pass, the KV cache gets extended with the new token's key-value pairs.

Let me trace through how the KV cache evolves: after the initial prefill, it holds the prompt's keys and values, then with each decode step it grows by one token. So kv_real_len tracks the actual physical length of the cache for each slot, which is always the prompt length plus however many tokens we've generated so far. Now I'm thinking through the batched decode algorithm...

Implementing batched decode step... Now I'm stacking the KV caches with padding to align them, running a single batched forward pass through the model with the collected tokens and attention masks, then splitting the returned KV caches back into their respective slots. Writing the KV cache stacking function...

Now I'm tackling the core challenge: with a batched forward pass, I can't register N separate steering hooks since there's only one forward execution. Instead, I need a single merged hook that applies SLERP independently for each slot in the batch, using that slot's alpha value from its state. Since the hidden states come through as [N, 1, d], I can apply the steering row by row within one hook call.

Implementing the batched steering hook...

For the spherical rotation case, I apply the same injection logic but with a different transformation. I need to handle both fixed alpha values in Continuous mode and dynamic alpha from slot state, then update the hidden states in place. The output structure needs to be preserved—if it's a tuple, I reconstruct it with the modified hidden tensor, otherwise just return the hidden tensor.

For the StateMonitor, I'm iterating through each slot and calling its monitor with the corresponding logits slice, which lets each slot track its own steering effectiveness independently.

Now I'm calculating the memory footprint of stacking all N KV caches with padding—with 16 slots, 8000 token max length, and standard head dimensions across 32 layers, this comes to roughly 2GB per round, which is manageable on larger GPUs but worth monitoring.

The stacking operation itself doesn't permanently duplicate memory since it's just a view during the forward pass, and once we extract the per-slot results afterward, the stacked tensor gets freed. I'm also reconsidering the hook mechanism—instead of registering it per-step, I could keep it active for the entire batched decode with a mutable reference to the active slots list, letting the main loop update which slots are currently processing.

Now I'm thinking through the complete revised implementation and what changes are needed to the main generation loop.

Structuring the main decode loop around three key operations: first, running the StateMonitor serially across all active slots to update their alpha values, then executing a single batched decode step, and finally extracting logits and sampling tokens before handling slot completion and refilling. The hook registration stays constant throughout, but I need to ensure it reads the alpha values that StateMonitor just computed in the current step.

Looking at the timing more carefully—StateMonitor runs after each forward pass to compute alpha for the next step, so the sequence is: forward produces logits, StateMonitor processes those logits to set alpha, then the next forward's hook reads that alpha during its execution.

In batched mode, this scales naturally: each slot maintains its own alpha state that gets updated after sampling, then the batched forward reads all the alphas simultaneously through the hook. When a new slot completes and gets added, it goes through its own prefill phase independently before joining the batched iterations.

The key insight is that slot removal happens after the batched forward, so a newly pending slot can immediately start its prefill while the current batch continues—then it joins the active pool for the next step with its own initialized state and padded key-values.

Defining the continuous batching algorithm... Setting up the main decode loop where I prefill initial slots, then iterate through batched forward passes while updating active slots based on completion status and applying per-slot filtering to the logits.

Still writing decode logic... I'm verifying the logic flow: _slot_to_result reads slot.input_ids before we call _cleanup_slot which nullifies it, so that's correct. But I'm realizing there's a potential issue with how the attention mask gets rebuilt during the batched decode step—it's based on the KV cache length from slot.past_key_values, and I need to trace through what happens to that after the prefill phase.

When we run the first batched decode step with a max KV length M, all slots' KV caches get padded to length M+1, but slots that were originally shorter than M now have garbage values in the early positions that should be masked out by the attention mask. The problem is when I extract a slot's KV cache afterward, I'm not properly accounting for which positions are actually valid versus padded.

For the next round, I need to track the real content length separately because the slot's KV will have length M+1 but only the last real_len_i + 1 positions are actually meaningful—the rest is garbage that needs to be masked out in the attention mask. So I do need to maintain slot.kv_real_len as a separate field to know exactly which positions to attend to.

After each decode step, I increment the real length and construct the attention mask by calculating how much padding exists, then creating a mask that zeros out the padded positions while allowing attention to the real content plus the newly generated token.

But I'm realizing the slots don't all have the same physical KV length—newly prefilled slots have their prompt length while existing slots have grown by one token each round. So I need to track the real KV length separately from the physical length to compute the mask correctly. Let me redesign the slot dataclass to explicitly store this real length, set it to the input length during prefill, and increment it after each decode step.

Defining the slot structure... I'm realizing the mask calculation was off—after each batched decode step, all slots' physical KV lengths become equal to max_physical + 1, but their kv_real_len only increments by 1, so the padding for each slot remains constant across steps.

The tricky part is when a newly prefilled slot joins the batch: it starts with no padding since its physical length equals its real length, but when stacked with existing slots that have grown to M+1, it needs to be padded by (M+1 - prompt_len) positions.

Now I'm thinking through the implementation more carefully. The key insight is that kv_real_len tracks the actual number of real tokens in the cache, while the attention mask always pads the front with zeros for garbage positions and ones for real tokens plus the new decode step. After prefill, kv_real_len equals the input length, and it increments with each decode step.

For the hook in batched mode, it fires once per forward pass with hidden states shaped as batch of 1-token sequences. I need to apply per-slot SLERP using each slot's alpha value, and the critical part is ensuring that active_slots_ref[0] matches the batch dimension order since slots are always stacked consistently.

Writing the hook function... I'm realizing there's a design issue here—for Continuous mode, the alpha value is already baked in during slot initialization, so I can just read it directly from slot.state.alpha. The Dynamic_Spherical mode handles updates through StateMonitor, and this is all already managed by _build_slot_components. Now I need to verify that the batched approach actually works correctly with the spherical_rotate function, which expects hidden states in a specific shape.

I could optimize this further by batching the entire SLERP operation across all N slots at once—collecting all alpha values into a single tensor and applying the rotation to all hidden states simultaneously rather than processing them individually.

Now I'm mapping out the implementation steps: I need to add a kv_real_len field to track the actual key-value length in each slot, update the prefill logic to set this value, replace the sequential decode loop with a batched version, and create a new steering hook that handles the batched operations. Writing the refactored code...

Actually, I'm realizing there's a synchronization issue: if the batched hook gets registered before prefills start, it might see partial slot states during concurrent prefill and decode operations. I need to ensure prefills complete atomically before the decode loop begins using the shared hook, or handle the hook registration more carefully to avoid race conditions.

Let me think through the edge cases: single active slot should work fine as a batch-of-one, slots with matching KV lengths avoid unnecessary padding, and new slots joining mid-decode need proper initialization.

After each batched decode step, I need to increment kv_real_len for all active slots in the per-slot loop that follows. Now I'm planning the actual diff—the main changes go into run_experiment.py where I'll add the kv_real_len field to the slot class, refactor the prefill to remove per-slot hook registration and initialize this field properly, and remove the old hook setup code.

Actually, I should keep _register_slot_hook itself but just stop calling it from _prefill_prompt—instead I'll register hooks once at the start of the main generation loop. For memory efficiency after the batched decode, I'm cloning the returned KV caches to free the original tensors...

Handling memory cleanup after splitting KV caches... large stacked tensor with .clone() creating independent copies for each slot. Without cloning, I'd just have views into the stacked tensor, which would keep it alive longer in memory. Either way the total memory usage is roughly equivalent—with cloning, both the stacked tensor and individual tensors exist simultaneously before the stacked one gets freed, whereas without cloning the stacked tensor lingers while views reference it.

I'll go with .clone() for cleaner memory semantics. Now I need to outline the actual code changes: adding a kv_real_len field to track the real length of each slot's KV cache, removing the hook-based approach entirely, and introducing helper functions to stack the padded KV caches and create a batched steering hook that processes all active slots at once instead of looping through them sequentially.

The main refactor is restructuring run_continuous_batching_generation to register a single shared hook upfront, handle prefill without per-slot hooks, then use a batched decode step in the main loop where all active slots are processed together before doing per-slot sampling and completion checks. I'll also simplify _cleanup_slot to skip the hook removal logic since there won't be per-slot hooks anymore.

Writing the code changes... to be efficient.

Let me look at the current lines of the file first to plan the edit more precisely.

The _Slot dataclass is at lines 473-493 approx. _build_slot_components is at 496-541. _register_slot_hook is at 544-558. _slot_to_result is at 561-589. _safe_score_range_clean is at 592-601. run_continuous_batching_generation is at 604-829. _cleanup_slot is at 832-840.

I need to modify _Slot to add a kv_real_len field, remove _register_slot_hook and replace it with a new batched steering hook creator, keep the other helper functions, add two new functions between _safe_score_range_clean and the main generation loop, completely rewrite run_continuous_batching_generation, and strip out the hook handle logic from _cleanup_slot. I'll take a targeted approach, keeping existing helpers where they work and only rewriting what needs to change.