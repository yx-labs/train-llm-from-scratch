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
- `CH0-03_vector_rail.md`: composes certified ScalarCell instances into ordered `vector[C]`; order is behavior, not just shape.
- `CH0-04_dot_product.md`: multiply + reduce behavior proof; forbids prebuilt DotProduct.
- `CH0-05_matrix_board.md`: two vector columns become `matrix[C,O]`; column order and axis orientation are tested.
- `CH0-06_matrix_multiply.md`: MatMulGate consumes C and emits O; requires TensorBox/MatrixStruct thinking and numeric reference.
- `CH0-07_tensor_box.md`: token vectors become `tensor[B,T,C]`; distinguishes T from C before MatMul/Linear.
- `CH1-01_splitter.md`: raw text becomes exact `string_piece[T]`; includes punctuation behavior and PieceBuffer probe.
- `CH1-02_merge_forge.md`: adjacent pair scan + merge rule table + merge apply; tests exact merged pieces.
- `CH1-04_vocab_lookup.md`: token pieces become integer IDs through table lookup and fallback; tests ID sequence behavior.
- `CH1-08_tokenizer_component.md`: composes splitter/merge/vocab/pad/mask into IDs + attention mask; tests EOS, padding mask, and token budget.
- `CH2-02_embedding_lookup.md`: token IDs gather rows from embedding table; tests row behavior, not just `[B,T,C]`.
- `CH3-02_linear_module.md`: Linear is built from certified MatMulGate + BroadcastRail + AddGate; no prebuilt Linear.
- `CH4-03_qk_score.md`: QKScore requires K transpose + certified MatMulGate + ScoreBoard + CellTrace; includes T==D hidden trap.
- `CH4-05_causal_mask.md`: builds future-mask matrix from query/key index grids; tests mask values and orientation.

Self-check command:

```text
rg "任务不是背公式|观察预制输入节点|添加或修复|公开认证不要求玩家手写完整 tensor|认证通过：同一张图|输出必须保持 dtype" <rewritten-core-docs>
```

Result: no matches in the rewritten core set.

Directory-wide template scan still reports 59 roadmap placeholders. These are intentionally not promoted to playable content yet.

## Implementation Notes

- Existing handcrafted MVP0.1 levels remain the high-fidelity examples for the earliest graph-construction arc.
- Generic source -> target component -> contract levels are no longer considered shippable playable content.
- Roadmap levels may stay visible on the challenge map, but should not be selectable until they receive a real internal build design.
- Do not promote a roadmap document to playable until it defines: player-built internal graph, forbidden target shortcut, behavior assertion, hidden mutation, certification controls, and probe-vs-component distinction.
- Next recommended rewrites: `CH3-00_linear_core.md`, `CH3-01_bias_add.md`, `CH4-04_scale.md`, `CH4-06_mask_apply.md`, `CH4-07_softmax.md`, and `CH4-08_weighted_sum.md`.

## Remaining Risk

Most non-playable roadmap documents still contain the original generated template. They are acceptable only as route placeholders. They should not be treated as final design or implementation specs until rewritten with unique mechanics.
