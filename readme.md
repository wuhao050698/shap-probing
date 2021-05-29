## Generalization of the Perturbed Masking

Perturbed Masking use the mask to calculate the
impact of each pair in the sentence. We use shapley-based
method to extend the Perturbed Masking to get a new probing
task called shap probing.

---
### environment
```
tqdm
numpy
transformers==2.5.1
torch==1.4.0
```

### start
- run_shap.py: code to run the dependency parsing task 

```
python run_shap.py
```

- SHAP_probing.ipynb: code about Visualization 