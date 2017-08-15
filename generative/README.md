# Generating Sketches from Images

Can we use visual semantic information captured by modern deep neural networks to "train" a model to sketch interpretable images?

### Breakdown of steps
1. Create a feature extractor.
2. The "loss" between a sketch canvas and the natural image is the cosine similarity between the deep net embeddings and a weighted cost for classification error on the sketch canvas.
3. We start from a blank canvas; we will iteratively sample segments to compose a sketch. To sample a segment, sample a start coordinate and an end coordinate. There will always be a prior point; from a blank canvas, we start at the center; for all future points, we use the previous end point as the new start point and sample a new end point using a 2D gaussian distribution.
4. We concurrently sample a categorical variable: draw, move, or stop. We need to somehow assign probabilities to these; p(stop) should increase over time.
5. Repeat until stop.

### Example
```
python pix2sketch.py
    --imagepath ./data/car_natural.jpg
    --distractdir ./data/distractors
    --sketchdir ./sketch/  # replace me with wtv folder you use locally
    --n_samples 100
    --n_iters 10
```