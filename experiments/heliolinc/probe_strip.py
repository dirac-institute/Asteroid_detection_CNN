"""Probe: (1) how many DP2 diffim panels overlap the NEO strip in a July window, and
(2) can I map a sky point -> detector via visit_detector_region. Validates the targeted-manifest plan."""
from lsst.daf.butler import Butler
import lsst.sphgeom as sph
import math
b = Butler("dp2_prep")
STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"

# NEO strip box RA 295-320, Dec -25..-15 as a sphgeom polygon
def box(ra0, ra1, dec0, dec1):
    pts = [sph.UnitVector3d(sph.LonLat.fromDegrees(ra, dec))
           for ra, dec in [(ra0, dec0), (ra1, dec0), (ra1, dec1), (ra0, dec1)]]
    return sph.ConvexPolygon(pts)
strip = box(295, 320, -25, -15)

# (1) diffim panels overlapping the strip in a 1-day probe window
refs = list(b.registry.queryDatasets("difference_image", collections=STAGE4,
            where="instrument='LSSTCam' AND visit.day_obs>=20250703 AND visit.day_obs<20250705 "
                  "AND visit_detector_region.region OVERLAPS my_region",
            bind={"my_region": strip}))
visits = set(r.dataId["visit"] for r in refs)
print(f"diffim panels overlapping strip, 2-day probe: {len(refs)} panels, {len(visits)} visits")

# (2) point -> detector for the first visit
if refs:
    v = sorted(visits)[0]
    recs = list(b.registry.queryDimensionRecords("visit_detector_region", instrument="LSSTCam", where=f"visit={v}"))
    print(f"visit {v}: {len(recs)} detector regions")
    # test a point near strip center
    pt = sph.LonLat.fromDegrees(307.0, -20.0)
    hit = [r.detector for r in recs if r.region.contains(sph.UnitVector3d(pt))]
    print(f"detectors containing (307,-20): {hit[:5]}")
