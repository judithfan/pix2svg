import numpy as np
from svgpathtools import svg2paths, Path, Line, wsvg
from rdp import rdp
from svg.path import parse_path


"""
Helper functions used in conversions. 
Some functions from: 
https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/utils.py
"""

def read_svg_file(filename):
    paths, attributes =  svg2paths(filename)
    return paths

def lines_to_strokes(lines):
    """Convert polyline format to 3-stroke format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]

def to_big_strokes(stroke, max_len=250):
    """Make this the special bigger format as described in sketch-rnn paper."""
    result = np.zeros((max_len, 5), dtype=np.float64)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result

def strokes_to_lines(strokes):
    """Convert 3-stroke format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += strokes[i, 0]
            y += strokes[i,1]
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += strokes[i,0]
            y += strokes[i,1]
            line.append([x, y])
    return lines

def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
        if l == 0:
            l = len(big_stroke)
    result = np.zeros((l, 3), dtype=np.float64)
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


"""
Functions below here are full conversion functions from svg format
to either stroke format or from a stroke format back to svg format.
These functions should be imported and utilized, not the helper
functions above.
"""

def svg_to_stroke3(filename):
    paths = read_svg_file(filename)
    rdp_lines = []
    for path in paths:
        line_segs = []
        for point in path:
            # print point.start
            start = [np.real(point.start), np.imag(point.start)]
            end = [np.real(point.end), np.imag(point.end)]
            line_segs.append(end)
            line_segs.append(start)
        rdp_lines.append(rdp(line_segs, epsilon=2.8))
    return lines_to_strokes(rdp_lines)

def stroke3_to_svg(strokes, filename):
    lines = strokes_to_lines(strokes)
    paths = []
    for line in lines:
        path = Path()
        for i in range(1, len(line)):
            start = line[i][0] + line[i][1]*1j
            end = line[i-1][0] + line[i-1][1]*1j
            path.append(Line(start, end))
        paths.append(path)
    assert filename is not None
    wsvg(paths, filename= filename)

def svg_to_stroke5(filename):
    strokes = svg_to_stroke3(filename)
    return to_big_strokes(strokes, max_len = len(strokes) + 1)

def stroke5_to_svg(strokes, filename):
    stroke3_to_svg(to_normal_strokes(strokes), filename)


