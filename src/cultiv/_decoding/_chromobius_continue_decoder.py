import pathlib

import numpy as np
import sinter
import stim


class ChromobiusContinueDecoder(sinter.Decoder):
    """Chromobius, except failing to lift results in a False prediction instead of an error."""

    def decode_via_files(self, *, num_shots: int, num_dets: int, num_obs: int,
                         dem_path: pathlib.Path, dets_b8_in_path: pathlib.Path,
                         obs_predictions_b8_out_path: pathlib.Path,
                         tmp_dir: pathlib.Path) -> None:
        raise NotImplementedError()

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> sinter.CompiledDecoder:
        return CompiledChromobiusContinueDecoder(dem)


class CompiledChromobiusContinueDecoder(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel):
        import chromobius
        self.decoder = chromobius.compile_decoder_for_dem(dem)

    def decode_shots_bit_packed(
            self,
            *,
            bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        dets = bit_packed_detection_event_data
        result = np.zeros(shape=(dets.shape[0], 2), dtype=np.uint8)
        for k in range(len(dets)):
            try:
                result[k, 0] = self.decoder.predict_obs_flips_from_dets_bit_packed(dets[k])
            except:
                result[k, 1] = 1
        return result
