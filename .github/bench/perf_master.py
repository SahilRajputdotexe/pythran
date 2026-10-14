import importlib.util
import json
import sys
import timeit

from pythran import toolchain
from pythran.spec import spec_parser

ARRAYS = ("import numpy as np; np.random.seed(0); n = 1000000; "
          "a = np.random.rand(n) + 1; b = np.random.rand(n) + 1; "
          "c = np.random.rand(n) + 1; cols = 1000; rows = 500; "
          "image_list = [x % 10 + y % 20 for x in range(cols) for y in range(rows)]")

KERNELS = [
    ("cosine", """
#pythran export cosine(float[:], float[:])
import numpy as np
def cosine(a, b):
    return np.cos(a + b) * 2. + np.cos(a + b) * a
""", "cosine(a, b)", 10),
    ("cosine_once", """
#pythran export cosine_once(float[:], float[:])
import numpy as np
def cosine_once(a, b):
    return np.cos(a + b) * 2. + a
""", "cosine_once(a, b)", 10),
    ("ufunc_exp", """
#pythran export ufunc_exp(float[:], float[:])
import numpy as np
def ufunc_exp(a, b):
    return np.exp(-a * b) / (1. + np.exp(-a * b))
""", "ufunc_exp(a, b)", 20),
    ("exp_once", """
#pythran export exp_once(float[:], float[:])
import numpy as np
def exp_once(a, b):
    return np.exp(-a * b) / (1. + a)
""", "exp_once(a, b)", 20),
    ("elementwise", """
#pythran export elementwise(float[:], float[:], float[:])
def elementwise(a, b, c):
    return (a + b) * c + (a + b) / c
""", "elementwise(a, b, c)", 200),
    ("elementwise_long", """
#pythran export elementwise_long(float[:], float[:], float[:])
def elementwise_long(a, b, c):
    return ((a * b + c) * (a * b + c) + (a * b + c)) / (1. + (a * b + c))
""", "elementwise_long(a, b, c)", 200),
    ("sobelFilter", """
#pythran export sobelFilter(int list, int, int)
def sobelFilter(original_image, cols, rows):
    edge_image = list(range(len(original_image)))
    for i in range(rows):
        edge_image[i * cols] = 255
        edge_image[((i + 1) * cols) - 1] = 255

    for i in range(1, cols - 1):
        edge_image[i] = 255
        edge_image[i + ((rows - 1) * cols)] = 255

    for iy in range(1, rows - 1):
        for ix in range(1, cols - 1):
            sum_x = 0
            sum_y = 0
            sum = 0
            #x gradient approximation
            sum_x += original_image[ix - 1 + (iy - 1) * cols] * -1
            sum_x += original_image[ix + (iy - 1) * cols] * -2
            sum_x += original_image[ix + 1 + (iy - 1) * cols] * -1
            sum_x += original_image[ix - 1 + (iy + 1) * cols] * 1
            sum_x += original_image[ix + (iy + 1) * cols] * 2
            sum_x += original_image[ix + 1 + (iy + 1) * cols] * 1
            sum_x = min(255, max(0, sum_x))
            #y gradient approximatio
            sum_y += original_image[ix - 1 + (iy - 1) * cols] * 1
            sum_y += original_image[ix + 1 + (iy - 1) * cols] * -1
            sum_y += original_image[ix - 1 + (iy) * cols] * 2
            sum_y += original_image[ix + 1 + (iy) * cols] * -2
            sum_y += original_image[ix - 1 + (iy + 1) * cols] * 1
            sum_y += original_image[ix + 1 + (iy + 1) * cols] * -1
            sum_y = min(255, max(0, sum_y))

            #GRADIENT MAGNITUDE APPROXIMATION
            sum = abs(sum_x) + abs(sum_y)

            #make edges black and background white
            edge_image[ix + iy * cols] = 255 - (255 & sum)
    return edge_image
""", "sobelFilter(image_list, cols, rows)", 5),
    ("add3", """
#pythran export add3(float[:], float[:], float[:])
def add3(a, b, c):
    return a + b + c
""", "add3(a, b, c)", 200),
]


def build(name, code, tag):
    modname = name + "_" + tag
    module, _ = toolchain.generate_cxx(modname, code, spec_parser(code))
    with open("/tmp/cxx/%s.cpp" % modname, "w") as f:
        f.write(str(module))
    path = toolchain.compile_cxxcode(modname, str(module))
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


tag = sys.argv[1]
env = {}
exec(ARRAYS, env)
for name, code, call, number in KERNELS:
    env[name] = build(name, code, tag)
for name, code, call, number in KERNELS:
    best = min(timeit.repeat(call, number=number, repeat=7, globals=env))
    print(json.dumps({"tag": tag, "kernel": name, "t": best / number}))
