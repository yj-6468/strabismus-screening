from pathlib import Path

from strabismus_screen.types import PredictionWarning, ScreeningPrediction


def test_screening_prediction_serializes_slotted_warnings() -> None:
    prediction = ScreeningPrediction(
        video=str(Path("clip.mov")),
        test_type="Cover-Uncover",
        direction="exo_tropia",
        direction_probability={"exo_tropia": 0.8},
        clinical_class="tropia",
        clinical_probability={"tropia": 0.7},
        observable_in_clip="yes",
        horizontal_pd=20.0,
        horizontal_pd_interval=(15.0, 25.0),
        quality_score=0.9,
        warnings=[PredictionWarning("low_confidence", "Review this clip")],
    )
    assert prediction.to_dict()["warnings"] == [
        {"code": "low_confidence", "message": "Review this clip"}
    ]
