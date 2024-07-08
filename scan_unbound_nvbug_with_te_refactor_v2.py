import os
os.environ['XLA_FLAGS'] = "--xla_dump_to=nvbug_4675071_v2_dump --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_proto --xla_dump_hlo_as_html"

from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from transformer_engine.jax.cpp_extensions.base import register_primitive
from transformer_engine.jax.cpp_extensions.attention import FusedAttnFwdPrimitive, _FusedAttnRNGStateChecker
from transformer_engine.jax.cpp_extensions.base import register_primitive
from transformer_engine.jax.cpp_extensions.misc import get_padded_spec
from transformer_engine.jax.sharding import get_all_mesh_axes
#from transformer_engine_jax import NVTE_Bias_Type
#from transformer_engine_jax import NVTE_Mask_Type
#from transformer_engine_jax import NVTE_QKV_Layout
from transformer_engine.transformer_engine_jax import NVTE_Bias_Type
from transformer_engine.transformer_engine_jax import NVTE_Mask_Type
from transformer_engine.transformer_engine_jax import NVTE_QKV_Layout

def generate_cu_seqlen(actual_seqlen):
    """
    Generating cumsum seqlen for a batch
    """
    cu_seqlen = jnp.cumsum(actual_seqlen)
    cu_seqlen = jnp.hstack((0, cu_seqlen))
    return cu_seqlen


CP_AXIS = 'CP'


class FusedRingAttnFwdPrimitive(FusedAttnFwdPrimitive):
    """
    Fused Attention Forward Primitive
    """
    name = "te_fused_attn_forward"
    multiple_results = True
    impl_static_args = (7, 8, 9, 10, 11, 12)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def impl(q, k, v, bias, q_seqlen, kv_seqlen, seed, attn_bias_type, attn_mask_type, qkv_layout,
             scaling_factor, dropout_probability, is_training):
        assert FusedAttnFwdPrimitive.inner_primitive is not None

        q_cu_seqlen = generate_cu_seqlen(q_seqlen) / 8
        kv_cu_seqlen = generate_cu_seqlen(kv_seqlen) / 8

        output, softmax_aux, rng_state, _ = FusedAttnFwdPrimitive.inner_primitive.bind(
            q,
            k,
            v,
            bias,
            q_cu_seqlen,
            kv_cu_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training)
        return output, softmax_aux, rng_state

    @staticmethod
    def infer_sharding_from_operands(attn_bias_type, attn_mask_type, qkv_layout, scaling_factor,
                                     dropout_probability, is_training, mesh, arg_infos,
                                     result_infos):
        del attn_bias_type, attn_mask_type, scaling_factor
        del dropout_probability, is_training, result_infos
        del qkv_layout
        q_spec = get_padded_spec(arg_infos[0])
        k_spec = get_padded_spec(arg_infos[1])
        # q_spec = (...batch, q_seqlen, head, hidden)
        # k_spec = (...batch, kv_seqlen, num_gqa_groups, hidden)
        out_sharding = NamedSharding(mesh, PartitionSpec(*q_spec))
        softmax_aux_sharding = NamedSharding(
            mesh, PartitionSpec(*q_spec[:-3], q_spec[-2], q_spec[-3], None))
        rng_state_sharding = NamedSharding(mesh, PartitionSpec(get_all_mesh_axes(), None))
        return (out_sharding, softmax_aux_sharding, rng_state_sharding)

    @staticmethod
    def partition(attn_bias_type, attn_mask_type, qkv_layout, scaling_factor, dropout_probability,
                  is_training, mesh, arg_infos, result_infos):
        out_sharding = result_infos[0].sharding
        softmax_aux_sharding = result_infos[1].sharding
        rng_state_sharding = seed_sharding = NamedSharding(mesh,
                                                           PartitionSpec(get_all_mesh_axes(), None))
        arg_shardings = tuple([arg_i.sharding for arg_i in arg_infos[:-1]] + [seed_sharding])
        out_shardings = (out_sharding, softmax_aux_sharding, rng_state_sharding)


        partial_fa_impl = partial(FusedRingAttnFwdPrimitive.impl,
                                    attn_bias_type=attn_bias_type,
                                    attn_mask_type=attn_mask_type,
                                    qkv_layout=qkv_layout,
                                    scaling_factor=scaling_factor,
                                    dropout_probability=dropout_probability,
                                    is_training=is_training)
        def ring_attn_impl(q, k, v, bias, q_seqlen, kv_seqlen, seed):
            batch, q_seqlen, head, _ = q.shape
            output = jnp.zeros(q.shape).astype(q.dtype)
            softmax_aux = jnp.zeros((batch, head, q_seqlen, 1)).astype(jnp.float32)
            rng_state = jnp.zeros((2, 4)).astype(result_infos[2].dtype)

            def scan_kv_block(carry, idx):
                k_old, v_old, output_old, softmax_aux, rng_state = carry

                output, softmax_aux, rng_state = partial_fa_impl(q, k_old, v_old, bias, q_seqlen, kv_seqlen, seed)
                output_new = output_old + output

                k_new = jax.lax.ppermute(k_old, CP_AXIS, perm=[(i, (i + 1) % 8) for i in range(8)])
                v_new = jax.lax.ppermute(v_old, CP_AXIS, perm=[(i, (i + 1) % 8) for i in range(8)])
                return (k_new, v_new, output_new, softmax_aux, rng_state), None

            (k, v, output, softmax_aux, rng_state), _ = jax.lax.scan(scan_kv_block, init=(k, v, output, softmax_aux, rng_state), xs=jnp.arange(0, 8))

            return output, softmax_aux, rng_state

        return mesh, ring_attn_impl, out_shardings, arg_shardings


register_primitive(FusedRingAttnFwdPrimitive)


def fused_ring_attn_fwd(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, bias: jnp.ndarray,
                        q_seqlen: jnp.ndarray, kv_seqlen: jnp.ndarray, seed: jnp.ndarray,
                        attn_bias_type: NVTE_Bias_Type, attn_mask_type: NVTE_Mask_Type,
                        scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Wrapper for TE fused attention fwd, where query, key, value are seperated tensors
    Return BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    """
    checker = _FusedAttnRNGStateChecker()
    seed = checker.check_seed(seed, dropout_probability, is_training)

    if attn_bias_type == NVTE_Bias_Type.NVTE_NO_BIAS:
        assert bias is None
        bias = jnp.zeros(0, dtype=q.dtype)

    return FusedRingAttnFwdPrimitive.outer_primitive.bind(
            q,
            k,
            v,
            bias,
            q_seqlen,
            kv_seqlen,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training)


def ring_fused_attn(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                      q_seqlen: jnp.ndarray, kv_seqlen: jnp.ndarray):

    output, softmax_aux, _ = fused_ring_attn_fwd(
        q, k, v, None, q_seqlen, kv_seqlen, None,
        NVTE_Bias_Type.NVTE_NO_BIAS, NVTE_Mask_Type.NVTE_CAUSAL_MASK,
        1.0, 0.0, True)

    return output, softmax_aux

BATCH = 32
SEQLEN = 4096
HEAD = 64
HIDDEN = 32
DTYPE = jnp.bfloat16

def main():
    key = jax.random.PRNGKey(1124)
    q_key, k_key, v_key, key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (BATCH, SEQLEN, HEAD, HIDDEN), dtype=DTYPE)
    k = jax.random.normal(k_key, (BATCH, SEQLEN, HEAD, HIDDEN), dtype=DTYPE)
    v = jax.random.normal(v_key, (BATCH, SEQLEN, HEAD, HIDDEN), dtype=DTYPE)
    q_seqlen = jnp.full((BATCH,), SEQLEN, dtype=jnp.int32)
    kv_seqlen = jnp.full((BATCH,), SEQLEN, dtype=jnp.int32)

    devices = np.array(jax.local_devices())
    devices = devices.reshape((8,))
    with Mesh(devices, (CP_AXIS,)) as mesh:
        qkvo_sharding_ring = NamedSharding(mesh, PartitionSpec(None, CP_AXIS, None, None))
        seqlen_sharding = NamedSharding(mesh, PartitionSpec(None))

        q_seqlen = jax.device_put(q_seqlen, seqlen_sharding)
        kv_seqlen = jax.device_put(kv_seqlen, seqlen_sharding)

        q_ring = jax.device_put(q, qkvo_sharding_ring)
        k_ring = jax.device_put(k, qkvo_sharding_ring)
        v_ring = jax.device_put(v, qkvo_sharding_ring)

        jitted_ringl_fused_attn = jax.jit(ring_fused_attn,
                                           in_shardings=(qkvo_sharding_ring, qkvo_sharding_ring, qkvo_sharding_ring,
                                                         seqlen_sharding, seqlen_sharding),
                                           out_shardings=(qkvo_sharding_ring, seqlen_sharding))
        output_ring, softmax_aux_ring = jitted_ringl_fused_attn(
            q_ring, k_ring, v_ring, q_seqlen, kv_seqlen)



if __name__ == "__main__":
    main()



