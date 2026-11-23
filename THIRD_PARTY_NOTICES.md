# Third-Party Notices

TitanInfra is released under the BSD 3-Clause License (see `LICENSE`). It builds on
and includes code derived from the following open-source projects. Their
original copyright notices and license terms are preserved in the affected
source files and must be kept when redistributing.

## TorchTitan (BSD 3-Clause)

Copyright (c) Meta Platforms, Inc. and affiliates.
https://github.com/pytorch/torchtitan

TitanInfra depends on TorchTitan at runtime, and the per-family model code under
`titaninfra/models/<family>/` (`model.py`, `parallelize.py`, `sharding.py`,
`state_dict_adapter.py`, `config_registry.py`, and related files) is derived
from TorchTitan. Those files carry the Meta copyright header and remain under
the BSD 3-Clause License:

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Evaluation and Test Fixtures

Small evaluation fixtures under `tests/fixtures/datasets/` include records derived
from MATH-500, AIME, and FullStackBench-style examples. These records are used
only for testing data conversion and training behavior; no ownership of the
source datasets is claimed. Original dataset terms and attribution requirements
remain applicable. Generated MDS, tokenizers, and model checkpoints are test
inputs rather than TitanInfra-owned model assets.
