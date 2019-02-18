import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='HLS-v0',
    entry_point='gym_hls.envs:HLSEnv',
)

register(
    id='HLSMulti-v0',
    entry_point='gym_hls.envs:HLSMultiEnv',
)

