# PreCog NLP Task Submission

This is my submission for the PreCog NLP task on AI-generated text detection and adversarial testing. The project consists of four interconnected tasks that explore dataset construction, statistical analysis, detector training, interpretability analysis, and adversarial content generation.

## Task Overview

**Task 0: Dataset Construction**
Built a balanced dataset of human-written (Jane Austen, Elizabeth Gaskell) and AI-generated text (GPT-4, Claude, Gemma-2-27B) with 4000+ training and 1000+ test samples. Implemented stratified splitting to maintain class balance.

**Task 1: Statistical Fingerprinting**
Conducted comprehensive stylometric analysis across 12 features including lexical diversity, sentence complexity, and punctuation patterns. Found significant differences between human and AI text using Cohen's d effect size measurements.

**Task 2: Detector Training**
Developed three-tier detector system:
- Tier A: Baseline Naive Bayes achieving 93.5% accuracy
- Tier B: Ensemble methods with F1 scores of 0.982, 0.981, 0.980
- Tier C: DistilBERT with LoRA fine-tuning achieving over 99% accuracy

**Task 3: Interpretability Analysis**
Applied SHAP and Integrated Gradients to understand detector decision-making. Identified AI-specific linguistic patterns (AI-isms) and analyzed failure modes through false positive/false negative analysis. Found that detectors exhibit overconfidence on misclassifications.

**Task 4: Adversarial Generation**
Implemented a genetic algorithm to evolve AI-generated text that evades detection. Achieved fitness progression from 13.2% to 92.0% human-like score across 18 generations. Validated robustness through out-of-vocabulary testing and discovered that simple model outputs (Baby GPT) achieve near-perfect evasion, revealing fundamental detector limitations.

## Repository Structure

- **Notebooks**: Task_0_try3.ipynb, Task_1.ipynb, Task_2.ipynb, Task_3.ipynb, Task_4.ipynb
- **Reports**: PreCog_Task-Summary.pdf (Important insights), PreCog_Task_Report.pdf (Detailed report of metodology + findings) 
- **Data Files**: 
  - Raw text: data/raw/austen/, data/raw/gaskell/
  - Processed datasets: class1_human_data.json, class2_ai_story_paragraphse_try2/3.json, class3_ai_story_paragraphs_try2/3.json (present in Human, Class2, Class3 folders of data respectively)
  - Feature data: task1_fingerprint_features.csv, task1_metadata.json
  - GA results: task4_ga_results_fixed.csv
  - Glove embeddings can be accessed at: https://nlp.stanford.edu/projects/glove/
- **Models**: tier_c_detector/, tier_c_detector_try1/, tier_c_detector_try2/, tier_c_detector_try3/
- **Environment**: nlp_env/ (Python virtual environment)

## Setup and Installation

**Prerequisites:**
- Python 3.8+
- CUDA-compatible GPU (recommended for Task 2, Task 3, Task 4)
- Google GenAI API key (required for Task 0, Task 4)

**Environment Setup**

```bash
python -m venv nlp_env
source nlp_env/bin/activate
pip install torch transformers datasets
pip install scikit-learn numpy pandas matplotlib seaborn
pip install shap captum nltk textstat
pip install jupyter ipykernel
pip install google-generativeai peft accelerate
```

**API Configuration:**

For Task 0 and Task 4, you need a Google GenAI API key. Set it in the notebooks:
```python
genai.configure(api_key="YOUR_API_KEY_HERE")
```

## Running the Project

Execute notebooks in order:

**Task 0 - Dataset Construction:**
```bash
jupyter notebook Task_0_try3.ipynb
```
Runtime: 2-3 hours (depends on API rate limits)
Outputs: class1_human_data.json, class2_ai_data.json

**Task 1 - Statistical Analysis:**
```bash
jupyter notebook Task_1.ipynb
```
Runtime: 5-10 minutes
Outputs: task1_fingerprint_features.csv, task1_metadata.json

**Task 2 - Detector Training:**
```bash
jupyter notebook Task_2.ipynb
```
Runtime: 45-60 minutes (with GPU)
Outputs: tier_c_detector/ (DistilBERT + LoRA model)

**Task 3 - Interpretability:**
```bash
jupyter notebook Task_3.ipynb
```
Runtime: 30-45 minutes
Outputs: SHAP visualizations, error analysis results

**Task 4 - Adversarial Generation:**
```bash
jupyter notebook Task_4_improved.ipynb
```
Runtime: 4-6 hours (API-dependent)
Outputs: class3_ai_story_paragraphs.json, task4_ga_results_fixed.csv

## Dependencies

Core libraries used:

- **Machine Learning**: transformers, torch, scikit-learn, peft
- **Data Processing**: pandas, numpy, datasets
- **NLP Tools**: nltk, textstat
- **Interpretability**: shap, captum
- **Visualization**: matplotlib, seaborn
- **API Integration**: google-generativeai
- **Development**: jupyter, ipykernel, accelerate

## Methodology

**Dataset Construction:**
Collected 300+ paragraphs from Jane Austen and Elizabeth Gaskell novels, generated matching AI content using three frontier models (GPT-4, Claude-3.5-Sonnet, Gemma-2-27B) with author-specific prompts. Applied stratified train-test split maintaining 80-20 ratio.

**Statistical Analysis:**
Extracted 12 stylometric features including type-token ratio, sentence length statistics, syllable counts, punctuation patterns. Used Cohen's d to quantify effect sizes and identify discriminative features.

**Detector Development:**
Progressive complexity approach - started with Naive Bayes baseline, built ensemble methods combining multiple algorithms, culminated in DistilBERT with LoRA fine-tuning for parameter-efficient transfer learning. Used stratified k-fold validation to ensure robust evaluation.

**Interpretability:**
Applied model-agnostic (SHAP) and model-specific (Integrated Gradients) methods to extract word-level attributions. Analyzed misclassifications to identify failure modes and measured confidence gaps between correct and incorrect predictions.

**Adversarial Generation:**
Designed genetic algorithm with population of 20 paragraphs, tournament selection, crossover operators, and prompt-based mutation. Fitness function used ensemble detector outputs. Initial population generated with explicit style diversity (academic, conversational, technical, persuasive, narrative) to ensure varied starting points. Validated results through out-of-vocabulary testing and systematic investigation of evasion mechanisms.

## Key Results

- Tier C detector: 99.5% accuracy, 0.995 F1 score on test set
- Statistical features: Cohen's d values ranging from 0.45 to 0.87
- AI-isms frequency: 1.92 to 3.71 per paragraph across different AI sources
- Genetic algorithm: 78.8 percentage point improvement over 18 generations
- Personal writing test: 0.9% human score (strong detection)
- OOV robustness: Wikipedia (1.9%), cooking instructions (0.8%), academic writing (0.9%), Reddit posts (8.8%)
- Baby GPT evasion: 99.5% human score despite grammatical errors
- Actual Austen test: 65.6% human score (detector uncertainty on real human text)

## Insights and Limitations

The project revealed that state-of-the-art detectors can be evaded through simple approaches rather than sophisticated adversarial techniques. The Baby GPT experiment demonstrated that low-perplexity text from weak models achieves better evasion than carefully evolved adversarial content, suggesting detectors rely on surface-level statistical patterns rather than semantic understanding.

Limitations include reliance on specific author domains (19th century literature), potential overfitting to training distribution, and inability to generalize to modern writing styles. The detector's uncertainty on actual Jane Austen text (65.6% confidence) indicates training corpus bias.

## What I Learned

This project provided hands-on experience with the full machine learning pipeline - from dataset construction through adversarial testing. Key technical skills developed include transfer learning with LoRA, interpretability analysis with SHAP/Captum, genetic algorithm design, and rigorous experimental validation. The systematic investigation of detector failures revealed important insights about the fragility of AI detection systems and the gap between statistical pattern matching and genuine understanding.

## References

- Hugging Face Transformers: https://huggingface.co/transformers/
- SHAP: https://github.com/slundberg/shap
- Captum: https://captum.ai/
- DistilBERT: Sanh et al., "DistilBERT, a distilled version of BERT"
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- Google GenAI: https://ai.google.dev/

