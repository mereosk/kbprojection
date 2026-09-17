# Silver-LEX pilot: evaluation report

Baseline: LangPro + built-in WordNet only, no injected relations. Augmented: baseline + silver LEX relations extracted from e-SNLI human explanations by the named model. Denominators exclude items where the baseline call itself failed (prover server errors).

## Summary

| Run | n | Baseline | Augmented | Delta | Flips | Regressions |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 198 | 37 (18.7%) | 42 (21.2%) | +5 | 5 | 0 |
| claude-sonnet-5 | 198 | 37 (18.7%) | 38 (19.2%) | +1 | 1 | 0 |

## gemini-3.1-flash-lite -- flip detail

### `esnli_test_3611`

- Premise: Three friends are talking.
- Hypothesis: Three people are hanging out.
- Human explanation: If three people are hanging out, they are talking.
- Relations offered: `isa_wn(friend, person) ;; isa_wn(talk, hang out)`
- Minimal sufficient subset (ablation result): `isa_wn(talk, hang out)`

### `esnli_test_4516`

- Premise: A woman in a green flowered dress is holding a container of yarn animals.
- Hypothesis: A lady in a dress has a container of yarn toys.
- Human explanation: Woman in a green flowered dress is a lady in a dress.
- Relations offered: `isa_wn(woman, lady) ;; isa_wn(yarn animal, yarn toy)`
- Minimal sufficient subset (ablation result): `isa_wn(yarn animal, yarn toy)`

### `esnli_test_4412` **(flagged: proof-hacking artifact, not a genuine lexical fact)**

- Premise: A boy in a yellow t-shirt and pink sweater talks on a cellphone while riding a horse through a crowd of people who are looking on.
- Hypothesis: A boy has a yellow t-shirt and pink sweater.
- Human explanation: If he is wearing the yellow shirt and pink sweater then he has them.
- Relations offered: `isa_wn(in, have)`
- Minimal sufficient subset (ablation result): `isa_wn(in, have)`
- Function-word (suspect) relations offered: `isa_wn(in, have)`

### `esnli_test_4753`

- Premise: A black dog is running through a marsh-like area.
- Hypothesis: A dog is running through a marshy area.
- Human explanation: A black dog can be commonly referred to as a dog, and "a marshy area"is a rephrasing of "a marsh-like area".
- Relations offered: `isa_wn(marsh-like area, marshy area)`
- Minimal sufficient subset (ablation result): `isa_wn(marsh-like area, marshy area)`

### `esnli_test_8609`

- Premise: Four people walk across stepping stones in a body of water.
- Hypothesis: Four people are standing up.
- Human explanation: In order for one to walk they must be standing up.
- Relations offered: `isa_wn(walk, stand up)`
- Minimal sufficient subset (ablation result): `isa_wn(walk, stand up)`

## claude-sonnet-5 -- flip detail

### `esnli_test_8609`

- Premise: Four people walk across stepping stones in a body of water.
- Hypothesis: Four people are standing up.
- Human explanation: In order for one to walk they must be standing up.
- Relations offered: `isa_wn(walk, stand up)`
- Minimal sufficient subset (ablation result): `isa_wn(walk, stand up)`

## Headline, adjusted for flagged artifacts

Across all runs: 6 total flips, of which 5 have no function-word relation in their minimal sufficient subset, and 1 does and should not be counted as evidence the method identified a genuine missing lexical fact.
