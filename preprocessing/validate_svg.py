import os
from glob import glob
import numpy as np

# purpose: loops through directories containing raw svg files to check whether they are properly closed with a </svg> tag. 

def list_files(path, ext='svg'):
    result = [y for x in os.walk(path)
              for y in glob(os.path.join(x[0], '*.%s' % ext))]
    return result

def get_last_line(path):
	fi=open(path, 'r')
	lastline = ""
	for line in fi:
		lastline = line
	return lastline

def replace_last_line(path):
	f = open(path, "wb")
	f.seek(-len(os.linesep), os.SEEK_END) 
	f.write("</svg>" + os.linesep)
	f.close() 

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='svg')

    args = parser.parse_args()
    all_paths = list_files(args.data_dir)
    print len(all_paths)

    broken_files = []

    for p in all_paths:
    	lastline = get_last_line(p)
    	if lastline not in ["</svg>\n", "</svg>"]:	    	
	    	if len(lastline)>len("</svg>"):
	    		print p + ' too many </svg> tags or some other issue, check these manually...  ' + lastline
	    		broken_files.append(p)
	    	else:
		    	print p + ' missing </svg> tag, adding now...  ' + lastline
		    	with open(p, "a") as myfile:
		    		myfile.write("</svg>")

	with open("list_broken_files.txt", "w") as output:
	    output.write(broken_files)	
