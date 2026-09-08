#!/bin/bash
for e in ~/anaconda3/envs/*/bin/python; do
  echo "== $e =="
  "$e" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)" 2>&1 | head -2
  "$e" -c "import spconv; print('spconv ok', spconv.__version__)" 2>&1 | head -2
done
