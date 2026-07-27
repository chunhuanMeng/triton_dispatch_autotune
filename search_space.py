"""
Search space definition and config validation for Triton GEMM dispatch autotune.
Config is 5-dimensional: (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps).
Templates are tested independently — config does NOT include template.
"""
import itertools
from dataclasses import dataclass

# ═══ Search Space Candidates ═══
BLOCK_M_CANDIDATES = [4, 8, 16, 32, 64, 128, 256]
BLOCK_N_CANDIDATES = [32, 64, 128, 256, 512]
BLOCK_K_CANDIDATES = [32, 64, 128, 256]
NUM_STAGES_CANDIDATES = [1, 2, 3, 4]
NUM_WARPS_CANDIDATES = [4, 8, 16, 32]

# Templates: each template will be tested independently with the same config set
TEMPLATES = ["triton_mm", "bmg_persistent", "bmg_decode"]

# These are the template-specific config lists in Inductor's BMG heuristics.
BMG_PERSISTENT_CONFIGS = [
    (256, 128, 64, 3, 16),
    (256, 256, 64, 2, 32),
    (128, 512, 64, 2, 32),
    (256, 256, 128, 2, 32),
    (32, 256, 32, 2, 8),
    (8, 512, 32, 2, 8),
    (8, 512, 32, 2, 16),
]
BMG_DECODE_CONFIGS = [
    (32, 256, 32, 2, 8),
    (8, 512, 32, 2, 8),
    (8, 512, 32, 2, 16),
]

# These are the standard XPU Triton choices in Inductor.  Keep the worker
# search space identical to the corresponding heuristic instead of searching
# the generic Cartesian product and later producing configs that Inductor will
# never register.
STANDARD_INT8_CONFIGS = [
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
]
STANDARD_FLOAT_CONFIGS = [
    (32, 32, 16, 1, 2),
    (32, 32, 128, 2, 4),
    (32, 64, 32, 5, 8),
    (64, 32, 32, 5, 8),
    (64, 32, 128, 5, 4),
    (64, 64, 16, 2, 4),
    (64, 64, 32, 2, 4),
    (64, 64, 64, 3, 8),
    (64, 64, 128, 5, 4),
    (64, 128, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (64, 128, 64, 3, 4),
    (64, 128, 128, 4, 4),
    (128, 64, 32, 3, 4),
    (128, 64, 32, 4, 8),
    (128, 128, 32, 2, 8),
    (128, 128, 32, 3, 4),
    (128, 128, 64, 3, 4),
    (128, 128, 64, 5, 8),
    (128, 128, 128, 4, 8),
    (128, 256, 64, 4, 8),
]


# ═══ Config Dataclass (5-dim, no template) ═══
@dataclass(frozen=True)
class GemmConfig:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_stages: int
    num_warps: int

    @property
    def key(self):
        return (self.BLOCK_M, self.BLOCK_N, self.BLOCK_K, self.num_stages, self.num_warps)

    def to_dict(self):
        return {
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "BLOCK_K": self.BLOCK_K,
            "num_stages": self.num_stages,
            "num_warps": self.num_warps,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["BLOCK_M"], d["BLOCK_N"], d["BLOCK_K"], d["num_stages"], d["num_warps"])

    def __str__(self):
        return f"BM={self.BLOCK_M} BN={self.BLOCK_N} BK={self.BLOCK_K} NS={self.num_stages} NW={self.num_warps}"


@dataclass(frozen=True)
class DispatchConfig:
    """Six-dimensional dispatch key: template + five GEMM parameters."""

    template: str
    gemm: GemmConfig

    @property
    def BLOCK_M(self):
        return self.gemm.BLOCK_M

    @property
    def BLOCK_N(self):
        return self.gemm.BLOCK_N

    @property
    def BLOCK_K(self):
        return self.gemm.BLOCK_K

    @property
    def num_stages(self):
        return self.gemm.num_stages

    @property
    def num_warps(self):
        return self.gemm.num_warps

    @property
    def key(self):
        return (self.template, *self.gemm.key)

    def to_dict(self):
        return {"template": self.template, **self.gemm.to_dict()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["template"], GemmConfig.from_dict(data))

    def __str__(self):
        return f"{self.template}:{self.gemm}"


def is_valid_config(M, N, K, bm, bn, bk, ns, nw):
    """Check hardware constraints for a config on a given shape (template-agnostic)."""
    # Tile must have enough elements for warps (each warp handles 16×16=256 elements)
    if bm * bn < nw * 256:
        return False
    # BLOCK_K must not exceed K
    if bk > K:
        return False
    # BLOCK_M should not be way larger than M
    if bm > max(M * 8, 64):
        return False
    # BLOCK_N must not exceed N
    if bn > N:
        return False
    # Register pressure: acc elements per thread <= 128
    acc_regs = (bm * bn) // (nw * 16)
    if acc_regs > 128:
        return False
    return True


def is_good_config(M, N, K, bm, bn, bk, ns, nw):
    """
    VERY Conservative pruning to skip configs that are EXTREMELY unlikely to be optimal.
    This is a SAFE filter - it should NEVER filter out the optimal config.
    
    Pruning rules (very conservative - only remove obviously bad configs):
    1. BM extreme mismatch with M (only for VERY large M)
    2. BK extreme mismatch with K (only for VERY large K)
    3. BN extreme mismatch with N (only for VERY large N)
    """
    # Rule 1: BM should not be extremely mismatched with M
    # Only filter when BM is WAY too small for VERY large M (M >= 2048 and bm < 16)
    if M >= 2048 and bm < 16:
        return False
    # Very large BM for very small M wastes parallelism
    if M <= 2 and bm > 128:
        return False
    if M <= 4 and bm > 256:
        return False
    
    # Rule 2: BK should not be extremely mismatched with K
    # Only filter very small BK for VERY large K (K >= 8192)
    if K >= 8192 and bk < 64:
        return False
    # Very large BK for very small K is wasteful
    if K <= 128 and bk > 128:
        return False
    
    # Rule 3: BN should not be extremely mismatched with N
    # Only filter very small BN for VERY large N (N >= 8192)
    if N >= 8192 and bn < 64:
        return False
    # Very large BN for very small N wastes parallelism
    if N <= 256 and bn > 256:
        return False
    
    return True


def is_valid_for_template(M, N, K, config, template):
    """Additional template-specific validity check."""
    bm = config.BLOCK_M
    if template == "bmg_decode":
        # Decode template is only for small M
        if bm > 32:
            return False
    if template == "bmg_persistent":
        # Persistent needs at least 1 tile
        pass
    # triton_mm: no extra constraints
    return True


def generate_valid_configs(M, N, K):
    """Generate all valid 5-dim configs for a given shape."""
    configs = []
    for bm, bn, bk, ns, nw in itertools.product(
        BLOCK_M_CANDIDATES, BLOCK_N_CANDIDATES, BLOCK_K_CANDIDATES,
        NUM_STAGES_CANDIDATES, NUM_WARPS_CANDIDATES
    ):
        if is_valid_config(M, N, K, bm, bn, bk, ns, nw):
            configs.append(GemmConfig(bm, bn, bk, ns, nw))
    return configs


def generate_template_configs(M, N, K, template):
    """Return the same candidate list as Inductor for the requested template."""
    if template == "bmg_persistent":
        candidates = BMG_PERSISTENT_CONFIGS
    elif template == "bmg_decode":
        candidates = BMG_DECODE_CONFIGS
    else:
        return generate_valid_configs(M, N, K)

    # Inductor's BMG heuristics intentionally keep these candidates even when
    # a generic shape heuristic would reject a large BLOCK_M for small M.
    return [GemmConfig(*values) for values in candidates]


def template_config_keys(template, dtype="int8"):
    """Return the exact config keys registered by Inductor for a template."""
    if template == "triton_mm":
        values = STANDARD_INT8_CONFIGS if dtype == "int8" else STANDARD_FLOAT_CONFIGS
    elif template == "bmg_persistent":
        values = BMG_PERSISTENT_CONFIGS
    elif template == "bmg_decode":
        values = BMG_DECODE_CONFIGS
    else:
        return set()
    return {tuple(v) for v in values}


def generate_autotune_configs(M, N, K, dtype="int8"):
    """Union of the exact standard and BMG Inductor candidate sets."""
    configs = {}
    for template in TEMPLATES:
        for config in generate_template_configs(M, N, K, template) if template != "triton_mm" else [GemmConfig(*v) for v in (STANDARD_INT8_CONFIGS if dtype == "int8" else STANDARD_FLOAT_CONFIGS)]:
            if template == "triton_mm" and not is_valid_config(
                M, N, K, config.BLOCK_M, config.BLOCK_N, config.BLOCK_K,
                config.num_stages, config.num_warps
            ):
                continue
            configs[config.key] = config
    return list(configs.values())


def generate_good_configs(M, N, K):
    """Generate configs that pass both validity and conservative goodness check."""
    configs = []
    for bm, bn, bk, ns, nw in itertools.product(
        BLOCK_M_CANDIDATES, BLOCK_N_CANDIDATES, BLOCK_K_CANDIDATES,
        NUM_STAGES_CANDIDATES, NUM_WARPS_CANDIDATES
    ):
        if is_valid_config(M, N, K, bm, bn, bk, ns, nw):
            if is_good_config(M, N, K, bm, bn, bk, ns, nw):
                configs.append(GemmConfig(bm, bn, bk, ns, nw))
    return configs


# ═══ Shape List ═══
ALL_SHAPES = [
    # M=1 is decomposed by Inductor into elementwise/reduction kernels and
    # never reaches MM template autotuning. Keep these shapes documented but
    # exclude them from the template/config search and dispatch evaluation.
    # (1, 1024, 4096), (1, 1536, 2048), (1, 2048, 768), (1, 2048, 1408),
    # (1, 2816, 2048), (1, 3072, 4096), (1, 3584, 2560), (1, 4096, 1536),
    # (1, 4096, 4096), (1, 4096, 7168), (1, 4096, 14336), (1, 5120, 3584),
    # (1, 6144, 16384), (1, 7168, 2048), (1, 14336, 4096), (1, 28672, 4096),
    # (1, 32768, 6144),
    (2, 4096, 4096),
    (4, 1536, 2048), (4, 2048, 768), (4, 2048, 1408), (4, 2816, 2048),
    (4, 3072, 4096), (4, 3584, 2560), (4, 4096, 1536), (4, 4096, 4096),
    (4, 4096, 7168), (4, 4096, 14336), (4, 5120, 3584), (4, 6144, 16384),
    (4, 7168, 2048), (4, 28672, 4096), (4, 32768, 6144),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 1536, 2048), (32, 2048, 768), (32, 2048, 1408), (32, 2816, 2048),
    (32, 3072, 4096), (32, 3584, 2560), (32, 4096, 1536), (32, 4096, 4096),
    (32, 4096, 7168), (32, 4096, 14336), (32, 5120, 3584), (32, 6144, 16384),
    (32, 7168, 2048), (32, 28672, 4096), (32, 32768, 6144),
    (64, 4096, 4096),
    (128, 1536, 2048), (128, 2048, 768), (128, 2048, 1408), (128, 2816, 2048),
    (128, 3072, 4096), (128, 3584, 2560), (128, 4096, 1536), (128, 4096, 4096),
    (128, 4096, 7168), (128, 4096, 14336), (128, 5120, 3584), (128, 6144, 16384),
    (128, 7168, 2048), (128, 28672, 4096), (128, 32768, 6144),
    (256, 4096, 4096),
    (512, 1536, 2048), (512, 2048, 768), (512, 2048, 1408), (512, 2816, 2048),
    (512, 3072, 4096), (512, 3584, 2560), (512, 4096, 1536), (512, 4096, 4096),
    (512, 4096, 7168), (512, 4096, 14336), (512, 4096, 16384), (512, 5120, 3584),
    (512, 5120, 5120), (512, 5120, 20480), (512, 6144, 16384), (512, 7168, 2048),
    (512, 7168, 7168), (512, 7168, 28672), (512, 28672, 4096), (512, 32768, 6144),
    (1024, 1024, 4096), (1024, 4096, 4096), (1024, 4096, 14336),
    (1024, 4096, 16384), (1024, 5120, 5120), (1024, 5120, 20480),
    (1024, 7168, 7168), (1024, 7168, 28672), (1024, 14336, 4096),
    (2048, 1536, 2048), (2048, 2048, 768), (2048, 2048, 1408),
    (2048, 2048, 2048), (2048, 2816, 2048), (2048, 3072, 4096),
    (2048, 3584, 2560), (2048, 4096, 1536), (2048, 4096, 4096),
    (2048, 4096, 7168), (2048, 4096, 14336), (2048, 4096, 16384),
    (2048, 5120, 3584), (2048, 5120, 5120), (2048, 5120, 20480),
    (2048, 6144, 16384), (2048, 6656, 16384), (2048, 7168, 2048),
    (2048, 7168, 7168), (2048, 7168, 28672), (2048, 28672, 4096),
    (2048, 32768, 6144),
]


def get_shape_family(M, N, K):
    ai = 2 * M * N * K / (M * K + K * N + M * N * 4)
    return "compute_bound" if ai > 513 else "memory_bound"


def get_source_pattern(M, N, K):
    if M <= 4: return "llm_decode"
    elif M <= 64: return "small_batch"
    elif M <= 512: return "medium_batch"
    else: return "llm_prefill"


if __name__ == "__main__":
    total_raw = len(BLOCK_M_CANDIDATES) * len(BLOCK_N_CANDIDATES) * len(BLOCK_K_CANDIDATES) * \
                len(NUM_STAGES_CANDIDATES) * len(NUM_WARPS_CANDIDATES)
    print(f"Raw search space: {total_raw} configs (5-dim, template-agnostic)")
    for M, N, K in [(4, 28672, 4096), (2048, 4096, 4096), (1, 2048, 768)]:
        valid = generate_valid_configs(M, N, K)
        print(f"  Shape ({M},{N},{K}): {len(valid)} valid configs")
