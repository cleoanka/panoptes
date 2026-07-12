"""Mapping from detector-native labels to the canonical taxonomy.

COCO-trained detectors (YOLO26, RF-DETR, ...) share the 80-class COCO
vocabulary; custom fine-tuned models may emit richer labels (see
training/EGITIM.md). Everything funnels into
:class:`~panoptes.core.types.VehicleClass` here so downstream code never
branches on backend-specific names.
"""

from __future__ import annotations

from panoptes.core.types import VehicleClass

__all__ = ["DEFAULT_VEHICLE_CLASSES", "map_label"]

# Raw label (lower-case) -> canonical class. Covers COCO plus common
# custom-training vocabularies (Turkish + English variants).
_LABEL_MAP: dict[str, VehicleClass] = {
    # COCO
    "car": VehicleClass.CAR,
    "bus": VehicleClass.BUS,
    "truck": VehicleClass.TRUCK,
    "motorcycle": VehicleClass.MOTORCYCLE,
    "motorbike": VehicleClass.MOTORCYCLE,
    "bicycle": VehicleClass.BICYCLE,
    "person": VehicleClass.PERSON,
    # richer / custom vocabularies
    "van": VehicleClass.VAN,
    "minivan": VehicleClass.VAN,
    "minibus": VehicleClass.VAN,
    "pickup": VehicleClass.TRUCK,
    "pickup_truck": VehicleClass.TRUCK,
    "kamyon": VehicleClass.TRUCK,
    "kamyonet": VehicleClass.TRUCK,
    "tir": VehicleClass.TRUCK,
    "otobus": VehicleClass.BUS,
    "otomobil": VehicleClass.CAR,
    "araba": VehicleClass.CAR,
    "motosiklet": VehicleClass.MOTORCYCLE,
    "bisiklet": VehicleClass.BICYCLE,
    "emergency": VehicleClass.EMERGENCY,
    "ambulance": VehicleClass.EMERGENCY,
    "ambulans": VehicleClass.EMERGENCY,
    "fire_truck": VehicleClass.EMERGENCY,
    "police": VehicleClass.EMERGENCY,
    "police_car": VehicleClass.EMERGENCY,
}

#: Classes kept when ``DetectorConfig.classes`` is None (vehicles only —
#: PERSON is opt-in because sidewalks dominate many road scenes).
DEFAULT_VEHICLE_CLASSES: frozenset[VehicleClass] = frozenset(
    {
        VehicleClass.CAR,
        VehicleClass.VAN,
        VehicleClass.BUS,
        VehicleClass.TRUCK,
        VehicleClass.MOTORCYCLE,
        VehicleClass.BICYCLE,
        VehicleClass.EMERGENCY,
    }
)


def map_label(raw_label: str) -> VehicleClass | None:
    """Map a detector label to the canonical taxonomy.

    Returns None for labels that are not road users at all (traffic
    lights, dogs, ...) so backends can drop them early.
    """
    label = raw_label.strip().lower().replace(" ", "_")
    if label in _LABEL_MAP:
        return _LABEL_MAP[label]
    return None
