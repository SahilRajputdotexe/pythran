import importlib.util
import json
import sys
import timeit

from pythran import toolchain
from pythran.spec import spec_parser

ARRAYS = ("import numpy as np; np.random.seed(0); n = 1000000; "
          "a = np.random.rand(n) + 1; b = np.random.rand(n) + 1; "
          "c = np.random.rand(n) + 1")

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
