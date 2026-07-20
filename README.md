# Does domain-aware prompting help CLIP on a real sensor domain shift?

## What this is

A small zero-shot experiment with CLIP, not a new method and not a paper draft. The starting point was Prof. Nila Alipour Talemi's Style-Pro paper, which shows that telling CLIP what visual style an image is in (sketch, painting, cartoon) helps it generalize better across those styles. That specific result is already well covered in the literature on PACS/VLCS style shifts (Domain Prompt Learning, PromptStyler, DAPT all show basically the same thing), so running it again on PACS wouldn't really tell anyone anything new.

Instead this asks a narrower question: does the same trick still work when the domain shift isn't an art style but a different sensor, RGB vs thermal/infrared. That's a real domain gap that shows up in a lot of applied work (wildfire monitoring, remote sensing, night-time detection), and as far as I could find it hasn't really been tested with this exact setup.

## Data

Teledyne FLIR ADAS Thermal Dataset v2, free on Kaggle: `samdazel/teledyne-flir-adas-thermal-dataset-v2`. Vehicle-mounted camera rig, thermal and RGB frames, mostly time-synced, COCO-format bounding box annotations. Classes used: person, car, bike, truck, chosen because these are the categories with enough samples in both domains (FLIR uses "bike", not "bicycle", and rarer categories like dog don't have enough examples to test on). 150 crops per class per domain, 1,200 crops total, pulled from the bounding boxes with a bit of padding.

## Method

CLIP ViT-B/32, zero-shot, no training or fine-tuning anywhere. Four prompt conditions, tested on the same crops so the comparison is apples to apples:

- `bare` — just the class name, e.g. "car"
- `generic` — "a photo of a {class}", same wording for both domains
- `domain_aware` — one prompt that names the actual domain, "a thermal camera image of a {class}" for thermal, "a color photograph of a {class}" for rgb
- `domain_aware_ensemble` — a few different domain-aware phrasings per class, embeddings averaged together

RGB acts as the control here. Generic and domain-aware wording are nearly the same for RGB on purpose, so if a difference showed up there too, it would suggest the gain was just from prompt wording in general rather than from naming the domain specifically.

## Results

**Thermal (the actual test):**

| condition | accuracy | ECE |
|---|---|---|
| bare | 44.5% | 0.253 |
| generic | 44.0% | 0.308 |
| domain_aware | 50.7% | 0.092 |
| domain_aware_ensemble | 51.7% | 0.104 |

Domain-aware prompting improves thermal accuracy by 6.7 points over generic (McNemar p = 0.0004, bootstrap 95% CI on the difference: 0.032 to 0.102, entirely above zero). The ensemble version does slightly better still, +7.7 points, p < 0.0001.

**RGB (control):** generic 69.0%, domain_aware 70.7%, difference not significant (p = 0.23, CI crosses zero). No real effect where none was expected, which is what makes the thermal result credible rather than a fluke of prompt wording in general.

**Calibration improved more than accuracy did.** On thermal, generic prompting is badly overconfident (ECE 0.31, mean confidence 0.75 against 44% actual accuracy). Domain-aware brings that down to ECE 0.09, confidence much closer to actual accuracy. This shows up clearly in the reliability diagrams.

**The gain is not spread evenly across classes.** Most of it comes from truck (25% to 59% correct going from generic to domain-aware). Car is roughly flat. Person recall actually drops (97% to 84%), because under the generic prompt the model was collapsing almost everything ambiguous into "person" on thermal images, and domain-aware prompting breaks that collapse rather than genuinely getting better at recognizing people. Bike stays broken under both conditions (around 3 to 7% correct either way), domain-aware prompting does not fix it.

**Failure case check:** the handful of wrong predictions inspected all involved small objects, under 2% of the frame. Consistent with what you'd expect, but six examples isn't a systematic claim, just a pattern worth noting.

**t-SNE check:** RGB and thermal embeddings land in clearly separate regions of the space with almost no overlap, confirming this is a real, substantial domain shift and not a trivial case.

## What this adds up to

Domain-aware prompting gives a real, statistically significant improvement in both accuracy and calibration on a genuine sensor domain shift, with no such effect on a matched control condition. The effect is moderate, not dramatic, and concentrated in specific classes (truck) rather than uniform. That's an honest, defensible finding either way this had turned out, this run happened to land on the positive side.

## How to run

1. Open a new Kaggle notebook, attach `samdazel/teledyne-flir-adas-thermal-dataset-v2` as a data source (accept the terms first if you haven't).
2. Turn on internet in notebook settings, needed once to pull the CLIP weights from Hugging Face.
3. GPU on (T4 is enough), run all cells top to bottom.
4. Everything gets written to `/kaggle/working/results/`, figures as PNGs, metrics as CSVs, and a run manifest as JSON.

## Known limitations

- FLIR ADAS is dashcam footage from one vehicle rig, one dataset, one camera setup, one geography. Not a general claim about RGB-thermal domain shift everywhere.
- The domain-aware prompt wording was written by hand, not learned or searched over. A different phrasing could change the result in either direction.
- Only one model tested (CLIP ViT-B/32). SigLIP or a newer VLM might behave differently.
- The improvement is class-specific (mostly truck), not uniform, and bike detection stays broken under both prompt conditions. Worth being upfront about this rather than only reporting the average.
- No hyperparameter search, one prompt set per condition, one seed.

## Files in this folder

- `style_aware_prompting_flir_RAN.ipynb` — the actual notebook as run on Kaggle, with all outputs saved inline
- `results/figures/` — the 6 figures the run produced
- `results/metrics/` — crop manifest, raw predictions, summary metrics, significance test results, all as CSV
- `results/run_manifest.json` — exact config this run used (model, classes, sample caps, seed)
- `README.md` — this file
