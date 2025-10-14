 COMPUTATIONAL BUDGET COMPARISON
  ─────────────────────────────────────────────────────────────────────────

  | Metric                    | FunSearch (GPU)     | CGP (CPU)           |
  |---------------------------|---------------------|---------------------|
  | Hardware                  | 4× H100 GPU         | 64 CPU cores        |
  | Number of runs            | 2                   | 15                  |
  | Wall clock time           | 144h (2×72h)        | 1080h (15×72h)      |
  | GPU hours                 | 576                 | 42 (equiv)          |
  | Candidates evaluated      | 0 (est)             | 785,053,269         |
  | Best score                | Projected           | 0.66968967          |
  | TFLOPs                    | 270.0 (est)         | 12.56               |
  | Energy (kWh)              | 403 (proj)          | 1037 (proj)         |

  NORMALIZED METRICS
  ─────────────────────────────────────────────────────────────────────────

  • CGP uses 0.1× more compute (GPU-equivalent)
  • FunSearch: ~25k candidates projected (LLM-guided)
  • CGP: 785,053,269 candidates (exhaustive search)