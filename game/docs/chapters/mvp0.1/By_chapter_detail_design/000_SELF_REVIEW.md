# MVP0.1 Chapter 0-9 Detail Design Self Review

## Review Scope

- Reviewed 74 per-level design documents under `By_chapter_detail_design`.
- Checked each level has: component target, concrete case, player action, challenge, certification variant, and pass criteria.
- Checked the sequence follows section four: Scalar -> Vector -> Matrix/Tensor -> MatMul/Linear -> Tokenizer -> Embedding -> Attention -> Transformer -> Loss -> Training -> Generation.

## Findings

Feedback pass 1 found a blocking design issue: most later documents were structurally complete but still template-like. They described adding the target component itself instead of designing a unique internal construction challenge.

Current policy:

- Keep the 74-level chapter map visible as a roadmap.
- Only open high-fidelity component-build levels as playable.
- Rewrite playable level documents toward unique mechanisms: real case, internal graph, forbidden shortcut, error paths, behavior tests, and certification variants.
- Use `CH0-04_dot_product.md` as the first rewritten example.

## Rewritten Deep-Level Documents

The following documents have now been rewritten out of the generic template and can be used as implementation-quality specs:

- `CH0-00_wire_port.md`: Graph OS warm-up, explicitly not a component build; teaches wire direction, contract gate, and reference probe.
- `CH0-01_scalar_cell.md`: first reusable component; finite float32 scalar, editable literal, non-float rejection, no allclose-to-example trap.
- `CH0-02_scalar_add.md`: composes two certified ScalarCell outputs through ScalarAddGate; rejects naked-number bypass and hardcoded visible output.
- `CH0-03_vector_rail.md`: composes certified ScalarCell instances into ordered `vector[C]`; order is behavior, not just shape.
- `CH0-04_dot_product.md`: multiply + reduce behavior proof; forbids prebuilt DotProduct.
- `CH0-05_matrix_board.md`: two vector columns become `matrix[C,O]`; column order and axis orientation are tested.
- `CH0-06_matrix_multiply.md`: MatMulGate consumes C and emits O; requires TensorBox/MatrixStruct thinking and numeric reference.
- `CH0-07_tensor_box.md`: token vectors become `tensor[B,T,C]`; distinguishes T from C before MatMul/Linear.
- `CH0-08_typed_tensor.md`: adds dtype as a real contract; int token IDs, bool masks, and float hidden values cannot be silently mixed.
- `CH0-09_axis_tensor.md`: binds axis names to tensor ranks and tests T/C swaps even when dimension lengths match.
- `CH1-01_splitter.md`: raw text becomes exact `string_piece[T]`; includes punctuation behavior and PieceBuffer probe.
- `CH1-02_merge_forge.md`: adjacent pair scan + merge rule table + merge apply; tests exact merged pieces.
- `CH1-00_text_input.md`: validates editable raw text as `raw_text[]`; preserves spaces/case and rejects non-string input.
- `CH1-03_vocab_table.md`: builds vocab rows with unique piece/id contracts and required `<pad>`/`<unk>` specials.
- `CH1-04_vocab_lookup.md`: token pieces become integer IDs through table lookup and fallback; tests ID sequence behavior.
- `CH1-05_token_buffer.md`: separates fixed context capacity from valid token length; preserves valid mask instead of premature padding.
- `CH1-06_padder.md`: fills EMPTY slots using vocab pad id and valid mask; catches hardcoded pad and real-token-id-zero traps.
- `CH1-07_attention_mask_builder.md`: derives padding attention mask from valid slots, not from token id values.
- `CH1-08_tokenizer_component.md`: composes splitter/merge/vocab/pad/mask into IDs + attention mask; tests EOS, padding mask, and token budget.
- `CH2-00_parameter_matrix.md`: tags matrices as named trainable parameters with seed/checksum and parameter identity.
- `CH2-01_embedding_table.md`: binds parameter rows to vocab ids and validates `[V,C]` row semantics before lookup.
- `CH2-02_embedding_lookup.md`: token IDs gather rows from embedding table; tests row behavior, not just `[B,T,C]`.
- `CH2-03_position_counter.md`: generates int position range `0..T-1` from T; rejects hardcoded visible windows.
- `CH2-04_position_embedding.md`: gathers position table rows by position ids, including shuffled-id hidden cases.
- `CH2-05_hidden_init.md`: broadcasts `[T,C]` position vectors over `[B,T,C]` token embeddings and adds behaviorally.
- `CH3-00_linear_core.md`: uses certified MatMulGate to build `hidden[B,T,C] @ weight[C,O]`, with column-level trace.
- `CH3-01_bias_add.md`: broadcasts `[O]` bias over `[B,T,O]`; catches T/O confusion and shape-only passes.
- `CH3-02_linear_module.md`: Linear is built from certified MatMulGate + BroadcastRail + AddGate; no prebuilt Linear.
- `CH3-03_activation.md`: constructs ReLU from zero broadcast + elementwise max; catches abs/leaky/reduce mistakes.
- `CH3-04_mlp_up_projection.md`: uses Linear with `O=4C` width policy, not hardcoded visible width.
- `CH3-05_mlp_down_projection.md`: returns activated `4C` stream to `C` through Linear and channel-return checks.
- `CH3-06_mlp_module.md`: composes Up -> ReLU -> Down into a full MLP with stage probes and end-to-end reference.
- `CH4-00_q_projection.md`: reuses Linear to create query role output `[B,H,T,D]`; separates role semantics from shape.
- `CH4-01_k_projection.md`: builds key projection with distinct Wk/bk and forbids premature K transpose.
- `CH4-02_v_projection.md`: builds value projection for later weighted readout; keeps value role separate from score inputs.
- `CH4-03_qk_score.md`: QKScore requires K transpose + certified MatMulGate + ScoreBoard + CellTrace; includes T==D hidden trap.
- `CH4-04_scale.md`: computes `1/sqrt(D)` from HeadWidthProbe; rejects hardcoded visible-case scale.
- `CH4-05_causal_mask.md`: builds future-mask matrix from query/key index grids; tests mask values and orientation.
- `CH4-06_mask_apply.md`: turns causal mask into additive score bias; distinguishes additive masking from multiplication.
- `CH4-07_softmax.md`: implements stable row-wise softmax over the key axis with row-sum and masked-probability checks.
- `CH4-08_weighted_sum.md`: applies probabilities to V through MatMulGate; tests Tq/Tk alignment and behavior.
- `CH4-09_attention_head.md`: composes Q/K/V, score, scale, mask, softmax, and weighted sum into a full single-head pipeline.
- `CH5-00_head_split.md`: computes `D=C/H`, checks divisibility, and reshapes `[B,T,C]` into `[B,H,T,D]` with element mapping proof.
- `CH5-01_parallel_heads.md`: maps SingleHeadAttention over H while preserving head isolation and mask broadcast.
- `CH5-02_head_concat.md`: transposes and flattens `[B,H,T,D]` back to `[B,T,C]` with concat-order probes.
- `CH5-03_output_projection.md`: applies square Linear output projection and checks residual width.
- `CH5-04_multi_head_attention.md`: composes Q/K/V, head split, parallel heads, concat, and output projection into MHA.
- `CH6-00_mean_variance.md`: computes channel-wise mean/variance over C only, including centered-square behavior tests.
- `CH6-01_layernorm.md`: builds LayerNorm from stats, eps, rsqrt, normalize, and affine gamma/beta.
- `CH6-02_residual_add.md`: enforces exact same-shape residual addition without implicit broadcast.
- `CH6-03_attention_sublayer.md`: builds pre-norm attention sublayer with original hidden on residual path.
- `CH6-04_mlp_sublayer.md`: builds pre-norm MLP sublayer and checks MLP internal 4C stage.
- `CH6-05_transformer_block.md`: composes AttentionSublayer then MLPSublayer with residual stream probes.
- `CH6-06_block_stack.md`: applies TransformerBlock sequentially over N layers with per-layer weights.
- `CH7-00_final_layernorm.md`: applies final LayerNorm with distinct final-norm parameters before LM head.
- `CH7-01_lm_head.md`: projects final hidden to vocab logits with `[C,V]` weight and vocab-width checks.
- `CH7-02_logits_board.md`: binds logits to vocab axis without changing values, enabling loss and sampling probes.
- `CH7-03_shift_targets.md`: slices `T+1` token windows into input/target next-token pairs.
- `CH7-04_cross_entropy_cell.md`: builds stable log-softmax, target gather, and negative log likelihood.
- `CH7-05_loss_reducer.md`: computes masked mean loss over valid target positions only.
- `CH8-00_dataset_stream.md`: turns tokenizer output over corpus shards into a single EOS-separated token stream.
- `CH8-01_window_sampler.md`: slices continuous `T+1` training windows with bounds checks.
- `CH8-02_batch_builder.md`: stacks windows into batch input/target tensors while preserving B/T axes.
- `CH8-03_forward_runner.md`: calls the assembled model graph to produce `[B,T,V]` logits with trace probes.
- `CH8-04_backward_trace.md`: seeds loss gradients and verifies every trainable parameter receives same-shape gradients.
- `CH8-05_optimizer.md`: performs an AdamW step with optimizer state, weight decay, and frozen-parameter checks.
- `CH8-06_checkpoint.md`: packages params, optimizer state, step, and config hash into restorable checkpoints.
- `CH9-00_prompt_encoder.md`: reuses TextInput and Tokenizer to encode generation prompts into ids and mask.
- `CH9-01_context_crop.md`: keeps the last T tokens and synchronized mask for fixed-context generation.
- `CH9-02_next_logits.md`: runs forward pass and gathers logits at the last valid context position.
- `CH9-03_sampler.md`: applies temperature, top-k, and seeded categorical sampling.
- `CH9-04_append_token.md`: appends sampled token ids and updates sequence length.
- `CH9-05_decode.md`: decodes ids through inverse vocab while preserving piece spaces and unknown policy.
- `CH9-06_tiny_chat_loop.md`: composes prompt encode, crop, next logits, sampler, append, decode into a repeatable generation loop.

Self-check command:

```text
rg <legacy-template-marker-regex> game/docs/chapters/mvp0.1/By_chapter_detail_design
```

Result: no matches across the rewritten document set.

Directory-wide template scan now reports 0 roadmap placeholders.

## Implementation Notes

- Existing handcrafted MVP0.1 levels remain the high-fidelity examples for the earliest graph-construction arc.
- Generic source -> target component -> contract levels are no longer considered shippable playable content.
- Roadmap levels now have real internal build designs, but should not be selectable until the matching implementation exists.
- Do not promote a roadmap document to playable until it defines: player-built internal graph, forbidden target shortcut, behavior assertion, hidden mutation, certification controls, and probe-vs-component distinction.
- Next recommended work: map these designs into playable graph levels in small batches, starting with the next currently locked route cluster.

## Remaining Risk

The document template debt is cleared. The remaining risk is implementation drift: most Chapter 5-9 designs are now valid design specs, but the game still needs matching handcrafted graph levels, module behavior, assertions, probes, and certification variants before those roadmap nodes can become playable.
