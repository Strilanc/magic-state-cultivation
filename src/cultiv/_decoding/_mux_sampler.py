import sinter

from ._chromobius_continue_decoder import ChromobiusContinueDecoder
from ._chromobius_gap_sampler import ChromobiusGapSampler
from ._pymatching_gap_sampler import PymatchingGapSampler
from ._desaturation_sampler import DesaturationSampler
from ._highlander_sampler import HighlanderSampler
from ._no_touch_decoder import NoTouchDecoder
from ._vec_intercept_sampler import VecInterceptSampler
from ._twirl_intercept_sampler import TwirlInterceptSampler


def sinter_samplers() -> dict[str, sinter.Sampler]:
    return {
        'highlander': HighlanderSampler(),
        'vec_intercept_t': VecInterceptSampler(turns=0.25, sweep_bit_randomization=False),
        'vec_intercept_z': VecInterceptSampler(turns=1, sweep_bit_randomization=False),
        'vec_intercept_s': VecInterceptSampler(turns=0.5, sweep_bit_randomization=False),
        'vec_intercept_t_twirl': VecInterceptSampler(turns=0.25, sweep_bit_randomization=False),
        'vec_intercept_z_twirl': VecInterceptSampler(turns=1, sweep_bit_randomization=False),
        'vec_intercept_s_twirl': VecInterceptSampler(turns=0.5, sweep_bit_randomization=False),
        'twirl_intercept_t': TwirlInterceptSampler(turns=0.25),
        'twirl_intercept_z': TwirlInterceptSampler(turns=1),
        'twirl_intercept_s': TwirlInterceptSampler(turns=0.5),
        'notouch': NoTouchDecoder(discard_on_fail=True),
        'notouch-hope': NoTouchDecoder(discard_on_fail=False),
        'chromobius-continue': ChromobiusContinueDecoder(),
        'chromobius-gap': ChromobiusGapSampler(),
        'desaturation': DesaturationSampler(),
        'pymatching-gap': PymatchingGapSampler(),
    }
