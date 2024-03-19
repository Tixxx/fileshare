import argparse
from functools import partial
import numpy as np
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map

import datetime

with_sharding_constraint = nn_partitioning.with_sharding_constraint

def test_fn(input, weight1, weight2):
    out = input @ weight1
    out = with_sharding_constraint(out, ('batch', 'seq_ag', 'mlp'))
    out = out @ weight2
    return out

def main():
    parser = argparse.ArgumentParser(description='Collective Matmul Unit Test')
    parser.add_argument("--dp", dest="dp", type=int, default=8)
    parser.add_argument("--tp", dest="tp", type=int, default=1)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--seq_len", dest="seq_len", type=int, default=2048)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=12288)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    assert(args.dp * args.tp == len(list(jax.devices())))
    assert(args.seq_len % args.tp == 0)
    assert(args.hidden_size % args.tp == 0)
    args.batch_size = args.batch_size * args.dp

    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key, 2)
    input = jax.random.uniform(key2, (args.batch_size, args.seq_len, args.hidden_size), dtype=dtype)
    weight1 = jax.random.uniform(key2, (args.hidden_size, args.tp*args.hidden_size), dtype=dtype)
    weight2 = jax.random.uniform(key2, (args.tp*args.hidden_size, args.hidden_size), dtype=dtype)

    mesh_shape = {'dp': args.dp, 'tp': args.tp}
    mesh = Mesh(np.array(jax.devices()).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))
    logical_axis_rules = (('batch', 'dp'), ('seq_rs', 'tp'), ('seq_ag', None), ('emb', None), ('mlp', 'tp'))

    pjitted_test_fn = pjit(partial(test_fn), out_shardings=PartitionSpec('dp', 'tp', None))

    if args.profile:
        import ctypes
        libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
        with mesh, nn_partitioning.axis_rules(logical_axis_rules):
            input = with_sharding_constraint(input, ('batch', 'seq_rs', 'emb'))
            weight1 = with_sharding_constraint(weight1, ('emb', 'mlp'))
            weight2 = with_sharding_constraint(weight2, ('mlp', 'emb'))
            for i in range(100):
                if i == 9:
                    libcudart.cudaProfilerStart()
                out = pjitted_test_fn(input, weight1, weight2)
            libcudart.cudaProfilerStop()
    else:
        with mesh, nn_partitioning.axis_rules(logical_axis_rules):
            input = with_sharding_constraint(input, ('batch', 'seq_rs', 'emb'))
            weight1 = with_sharding_constraint(weight1, ('emb', 'mlp'))
            weight2 = with_sharding_constraint(weight2, ('mlp', 'emb'))
            for i in range(100):
                out = pjitted_test_fn(input, weight1, weight2)

            start = datetime.datetime.now()
            iteration = 5
            for i in range(iteration):
                out = pjitted_test_fn(input, weight1, weight2)
            end = datetime.datetime.now()
            delta = (end - start).total_seconds() * 1000 / iteration
            print("Delta avg: {}".format(delta))

    return out

if __name__ == "__main__":
    main()

