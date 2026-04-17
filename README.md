<p align="center">
  <h1 align="center">Entropy-Gradient Grounding: Training-Free Evidence Retrieval <br> in Vision-Language Models</h1>
  <p align="center">
    <a href="">Marcel Gropl</a><sup>1*</sup>
    ·
    <a href="https://crepejung00.github.io/">Jaewoo Jung</a><sup>3*</sup>
    ·
    <a href="https://cvlab.kaist.ac.kr/members/faculty">Seungryong Kim</a><sup>3</sup>
    ·
    <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1</sup>
    ·
    <a href="https://sunghwanhong.github.io/">Sunghwan Hong</a><sup>1,2</sup>
  </p>
  <h4 align="center"><sup>1</sup>ETH Zurich, <sup>2</sup>ETH AI Center, <sup>3</sup>KAIST</h4>

  <p align='center'><sup>*</sup>Equal contribution</p>

  <h3 align="center"><a href="#">Paper</a> | <a href="https://entropy-gradient-grounding.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>


> We propose a **training-free, model-intrinsic grounding method** for vision-language models that uses uncertainty as supervision. Our approach backpropagates **entropy of the next-token distribution** to visual token embeddings, yielding relevance maps without auxiliary detectors or attention-map heuristics — enabling robust grounding on detail-critical and high-resolution settings across seven benchmarks and four VLM architectures.

### 🚀 What to Expect
- [x] Inference code for LLaVA. <br>
- [ ] Inference code for Qwen 2.5. <br>
- [ ] Inference code for InternVL 3.5. <br>
- [ ] Single image demo. <br>

## Installation

Our code is developed based on Python 3.10, PyTorch 2.1.2, and CUDA 12.1.

We recommend using [conda](https://docs.anaconda.com/miniconda/) for installation:

```bash
git clone https://github.com/merci00001/entropy-gradient-grounding.git
cd entropy-gradient-grounding

conda create -n egrounding python=3.10
conda activate egrounding

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Method Overview

Given a query, **Entropy-Gradient Grounding** frames visual grounding as test-time evidence retrieval:

1. **Entropy-gradient relevance maps** — compute the entropy of the model's next-token distribution and backpropagate it to visual token embeddings to produce a grounding signal without any auxiliary detectors or attention-map heuristics.
2. **Multi-evidence ranking** — extract and rank multiple coherent regions to support compositional and multi-clue queries.
3. **Iterative zoom-and-reground** — refine localization iteratively with a spatial-entropy stopping rule to prevent over-refinement.

## Inference

### LLaVA 1.6 (Mistral)

```bash
python inference_llava.py \
  --model-path liuhaotian/llava-v1.6-mistral-7b \
  --image-folder /path/to/images \
  --question-file /path/to/questions.jsonl \
  --answers-file /path/to/answers.jsonl \
  --conv-mode mistral_instruct \
  --is15 False
```

### LLaVA 1.5

```bash
python inference_llava.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --image-folder /path/to/images \
  --question-file /path/to/questions.jsonl \
  --answers-file /path/to/answers.jsonl \
  --conv-mode vicuna_v1 \
  --is15 True
```

The `--question-file` follows the standard [LLaVA JSONL format](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md), where each line contains an image filename and a query.

**Key arguments:**

| Argument | Description |
|---|---|
| `--model-path` | HuggingFace model ID or local path |
| `--question-file` | Input questions in LLaVA JSONL format |
| `--answers-file` | Output file for model predictions |
| `--conv-mode` | `mistral_instruct` for Mistral-based models, `vicuna_v1` otherwise |
| `--is15` | Set `True` for LLaVA 1.5, `False` for LLaVA 1.6 |
| `--to_run` | Number of disjoint regions to keep track of |

### Qwen 2.5 / InternVL 3.5

Coming soon.


## Citation

```bibtex
@article{gropl2025egg,
  title={Entropy-Gradient Grounding: Training-Free Evidence Retrieval in Vision-Language Models},
  author={Gropl, Marcel and Jung, Jaewoo and Kim, Seungryong and Pollefeys, Marc and Hong, Sunghwan},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgement

We thank the authors of [LLaVA](https://github.com/haotian-liu/LLaVA) for their excellent work and code, which served as the foundation for this project.
