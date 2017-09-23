import numpy as np
import os
import helpers as svg
import pickle
from scipy.misc import imread, imresize


# Run this code to build the data files appropriately
class DataBuilder():

    def __init__(self, path):

        def parseSketches(filepath):
            all_classes = {}
            all_strokes = []
            for p in os.listdir(filepath):
                if p == ".DS_Store": continue
                all_ex = []
                lookup_dict = {}
                counter = 0
                print filepath, p
                print "%s/%s/invalid.txt" % (filepath, p)
                with open("%s/%s/invalid.txt" % (filepath, p)) as f:
                    invalid = set([line[:-1] for line in f.readlines()])
                total = len(os.listdir("%s/%s" % (filepath, p)))
                for ex in os.listdir("%s/%s" % (filepath, p)):
                    if ex == ".DS_Store" or ex == "checked.txt" or ex == "invalid.txt": continue
                    if ex[:-4] in invalid:
                        print ("this one's bs bruh")
                        total -= 1
                        continue
                    print ("%s/%s/%s" % (filepath, p, ex))
                    strokes = helpers.svg_to_stroke5("%s/%s/%s" % (filepath, p, ex))
                    if strokes == None:
                        total -= 1
                        print ("Invalid")
                        continue
                    if (strokes.shape[0] > 200): continue
                    all_strokes.append(strokes.shape[0])
                    all_ex.append(strokes)
                    lookup_dict[counter] = ex[:ex.find("-")]
                    print ("%d / %d" % (counter, total))
                    counter += 1
                all_classes[p] = (all_ex, lookup_dict)
                break
            print (all_classes)
            np.save('all_sketches.npy', all_classes)
            print (max(all_strokes))
            return all_classes, max(all_strokes)


        self.sketches = parseSketches("%s/svgraw" % (path))

if __name__ == '__main__':
    build = DataBuilder('tiny')


