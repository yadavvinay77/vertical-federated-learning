"""
Real Homomorphic Encryption wrapper using TenSEAL.
If TenSEAL is not installed, fallback to mock HE.
"""

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False

from .mock_homomorphic import MockHEContext

class TenSEALHEContext:
    def __init__(self, params):
        # Initialize TenSEAL context with given params
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.get("poly_mod_degree", 8192),
            coeff_mod_bit_sizes=params.get("coeff_mod_bit_sizes", [60, 40, 40, 60])
        )
        self.context.generate_galois_keys()
        self.context.global_scale = params.get("global_scale", 2 ** 40)

    def encrypt(self, value):
        return ts.ckks_vector(self.context, value)

    def decrypt(self, enc_value):
        return enc_value.decrypt()

    def add(self, enc_val1, enc_val2):
        return enc_val1 + enc_val2

    def multiply(self, enc_val1, enc_val2):
        return enc_val1 * enc_val2

def get_he_context(use_real_he, params):
    if use_real_he and TENSEAL_AVAILABLE:
        return TenSEALHEContext(params)
    else:
        print("Using Mock Homomorphic Encryption (TenSEAL not available or disabled).")
        return MockHEContext(params)
