# Local data requirements

The course instructor supplied the original data. It is not distributed in this repository.

For an authorized local reproduction, place these files in this directory:

| File | Schema |
| --- | --- |
| `train.json` | Columns `reviews` (text) and `sentiments` (binary integer: 0 negative, 1 positive) |
| `test.json` | Column `reviews`; no sentiment labels |
| `submission_example.csv` | Columns `reviews` and `predictions`; review order must match `test.json` |

The original experiment expects 7,401 training rows and 1,851 test rows. The final output check explicitly requires 1,851 predictions. This is the course experiment runner, not a general-purpose dataset loader.

Raw inputs and generated predictions are ignored by Git. Do not add course data, review examples, personal identifiers, or fitted models to the public repository.
