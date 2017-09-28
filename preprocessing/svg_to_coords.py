import numpy as np
import os
import helpers as svg
import pickle
from scipy.misc import imread, imresize
import pandas as pd
import os
import pandas as pd

# Run this code to build the data files appropriately
class LineBuilder():

    def __init__(self, path):

        def parseSketches(filepath):
            all_classes = {}
            category_list = np.array(os.listdir(filepath))
            ind = np.where(category_list==args.start_from)[0][0]
            category_list = category_list[ind:]

            for p in category_list:
                curr_class = {}
                if p == ".DS_Store": continue
                all_ex = []
                lookup_dict = {}
                fpath = []
                counter = 0
                print filepath, p
                print "%s/%s/invalid.txt" % (filepath, p)
                with open("%s/%s/invalid.txt" % (filepath, p)) as f:
                    invalid = set([line[:-1] for line in f.readlines()])
                total = len(os.listdir("%s/%s" % (filepath, p)))
                for ex in os.listdir("%s/%s" % (filepath, p)):                    
                    if ex == ".DS_Store" or ex == "checked.txt" or ex == "invalid.txt": continue
                    if ex[:-4] in invalid: 
                        total -= 1
                        continue
                    print ("%s/%s/%s" % (filepath, p, ex))  
                    try:                  
                        strokes = svg.svg_to_stroke5("%s/%s/%s" % (filepath, p, ex))
                        if len(strokes) == 0:
                            total -= 1
                            print ("Invalid")
                            continue
                        if (strokes.shape[0] > 200): continue
                        all_ex.append(strokes)
                        lookup_dict[counter] = ex
                        fpath.append("{}/{}/{}".format(filepath, p, ex))
                        print ("%d / %d" % (counter, total))
                        counter += 1
                    except:
                        print "Issue with " + "%s/%s/%s" % (filepath, p, ex) 
                        pass

                all_classes[p] = (all_ex, fpath)
                curr_class[p] = (all_ex, fpath)
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                save_path = os.path.join(args.save_dir, 'coords_{}.npy'.format(p))                
                np.save(save_path, curr_class)
            all_save_path = os.path.join(args.save_dir, 'all_coords.npy')
            np.save(all_save_path, all_classes)        
            return all_classes

        self.sketches = parseSketches(path)

class SplineBuilder():

    def __init__(self, path):

        all_classes = sorted([i for i in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir,i))])
        assert len(all_classes)==125

        ind = np.where(np.array(all_classes)==args.start_from)[0][0]
        all_classes = all_classes[ind:]

        # loop through classes
        for c in all_classes:
            class_path = os.path.join(path_to_svg,c)
            all_sketches = [i for i in os.listdir(class_path) if i.split('.')[1]=='svg']
            
            # initialize
            start_x = []
            start_y = []
            c1_x = []
            c1_y = []
            c2_x = []
            c2_y = []
            end_x = []
            end_y = []
            stroke_num = []
            spline_num = []
            stroke_counter = 0
            spline_counter = 0

            class_name = []
            photo_name = []
            sketch_name = []

            path_to_invalid = os.path.join(class_path,'invalid.txt')
            with open(path_to_invalid) as f:
                invalid = set([line[:-2] for line in f.readlines() if line[0]=='n'])
            cat = class_path.split('/')[-1]
            # loop through sketches
            for i,s in enumerate(all_sketches):
                if i%100==0:
                    print 'Extracted {} of {} {} sketches...'.format(i, len(all_sketches), cat)
                if s in invalid:
                    print s + ' marked invalid, moving on...'
                else:
                    sketch_path = os.path.join(path_to_svg,c,s)
                    # read in paths and loop through to get strokes and spline segments
                    try:
                        paths = read_svg_file(sketch_path)        
                        for path in paths:
                            stroke_counter += 1
                            for point in path:
                                if type(point) == svgpathtools.path.CubicBezier:
                                    start_x.append(np.real(point.start))
                                    start_y.append(np.imag(point.start))   
                                    c1_x.append(np.real(point.control1))
                                    c1_y.append(np.imag(point.control1))
                                    c2_x.append(np.real(point.control2))
                                    c2_y.append(np.imag(point.control2))
                                    end_x.append(np.real(point.end))
                                    end_y.append(np.imag(point.end))
                                    spline_num.append(spline_counter)
                                    spline_counter += 1         
                                    stroke_num.append(stroke_counter)
                                    class_name.append(os.path.dirname(sketch_path).split('/')[-1])
                                    photo_name.append(os.path.basename(sketch_path).split('-')[0] + '.jpg')
                                    sketch_name.append(os.path.basename(sketch_path))   
                    except:
                        print 'Issue with ' + sketch_path + '... moving on.' 
                        
            df = pd.DataFrame([start_x,start_y,c1_x,c1_y,c2_x,c2_y,end_x,end_y, \
                               stroke_num,spline_num,class_name,photo_name,sketch_name]) 
            df = df.transpose()
            df.columns = ['start_x','start_y','c1_x','c1_y','c2_x','c2_y','end_x','end_y', \
                          'stroke_num','spline_num','class_name','photo_name','sketch_name']

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, cat + '.csv')
            df.to_csv(save_path)
        

def read_svg_file(filename):
    paths, attributes =  svg2paths(filename)
    return paths

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='spline_coords')
    parser.add_argument('--data_dir', type=str, default='svg')
    parser.add_argument('--start_from', type=str, default='airplane')
    parser.add_argument('--stroke_type', type=str, default='line')

    args = parser.parse_args()

    if args.stroke_type=='line':
        build = LineBuilder(args.data_dir)
    elif args.stroke_type=='spline':
        build = SplineBuilder(args.data_dir)


