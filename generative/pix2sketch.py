from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


if __name__ == '__main__':
    import os
    import argparse
    from PIL import Image

    import torch
    import torchvision.transforms as transforms
    from torch.autograd import Variable

    from beamsearch import SemanticBeamSearch


    parser = argparse.ArgumentParser(description="generate sketches")
    parser.add_argument('--image_path', type=str, help='path to image file')
    parser.add_argument('--distract_dir', type=str, help='directory to distractor image files')
    parser.add_argument('--sketch_dir', type=str, help='directory to store sketches')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='number of samples per iteration')
    parser.add_argument('--n_iters', type=int, default=20,
                        help='number of iterations')
    parser.add_argument('--stdev', type=float, default=15.0,
                        help='standard deviation for Gaussian when sampling')
    parser.add_argument('--patience', type=int, default=5,
                        help='once the informativity measure stops improving, wait N epochs before quitting')
    parser.add_argument('--beam_width', type=int, default=2,
                        help='number of particles to preserve at each timestep')
    parser.add_argument('--embedding_layer', type=int, default=-1,
                        help='-1|0|1|...|7|8')
    parser.add_argument('--embedding_net', type=str, default='vgg19', help='vgg19|resnet152')
    parser.add_argument('--distance_fn', type=str, default='cosine',
                        help='cosine|l1|l2')
    parser.add_argument('--fuzz', type=float, default=1.0,
                        help='hyperparameter for line rendering')
    args = parser.parse_args()

    # prep images
    natural = Image.open(args.image_path)
    distractors = []
    for i in os.listdir(args.distract_dir):
        distractor_path = os.path.join(args.distract_dir, i)
        distractor = Image.open(distractor_path)
        distractors.append(distractor)

    preprocessing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    # grab embeddings for the natural & distractor images
    natural = Variable(preprocessing(natural).unsqueeze(0))
    distractors = Variable(torch.cat([preprocessing(image).unsqueeze(0)
                                      for image in distractors]))

    explorer = SemanticBeamSearch(112, 112, 224, beam_width=args.beam_width,
                                  n_samples=args.n_samples, n_iters=args.n_iters,
                                  stdev=args.stdev, fuzz=1.0,
                                  embedding_net=args.embedding_net,
                                  embedding_layer=args.embedding_layer)

    natural_emb = explorer.vgg19(natural)
    distractor_embs = explorer.vgg19(distractors)

    for i in range(args.n_iters):
        sketch = explorer.train(i, natural_emb, distractor_items=distractor_embs)

    im = Image.fromarray(sketch)
    im.save(os.path.join(args.sketch_dir, 'sketch.png'))
