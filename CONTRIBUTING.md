# Contributing

**Drone 4-RF is a work in progress.** The deterministic pipeline
(Stages 1–5) is implemented and tested; the machine-learning subsystem
(Stage 6) is implemented but **ships untrained on purpose** - there is no
bundled model, because a model trained on someone else's RF environment
and presented as universal would be worse than no model at all.

Bug reports, hardware reports (other SoapySDR devices, other antennas),
and code contributions are all welcome. But the single most valuable
thing you can contribute is **calibration data**.

---

## Why calibration data is the bottleneck

Everything the scanner reports is relative to a *learned local
background*. The detectors, the burst/hop analytics and the fusion
weights were tuned against one site, one antenna and one HackRF. That is
enough to make the pipeline work; it is nowhere near enough to know how
the scores behave in a warehouse, a rural field, a dense apartment
block, or next to a different brand of drone.

Concretely, the project cannot yet answer:

- How separable are drone control links from Bluetooth/BLE hopping when
  the hop grid is only partially observed?
- Which motor/ESC comb spacings actually recur across airframes, versus
  being an artifact of one aircraft's ESCs?
- What false-alarm rate do the default thresholds produce across
  different RF environments?
- Do the fusion weights hold up at all, or are they overfitted to the
  single environment they were developed in?

Those are empirical questions. They need data from more than one site.

## What to share (in order of usefulness vs. risk)

### 1. Feature datasets - preferred, lowest risk

```sh
# after labelling some events (GUI buttons, or `drone4rf ml label`)
python -m drone4rf ml build-dataset --site <a-nickname> --out mydata.npz
```

This file contains **only abstract behavioural scores** - burst,
hopping, EMI, background-similarity, observation counts - plus your
labels. It contains **no IQ samples, no demodulated content, no absolute
frequencies, and no timestamps**. You can inspect exactly what is in it:

```sh
python -c "import numpy as np; d=np.load('mydata.npz'); print(d['feature_names'], d['X'].shape)"
```

Please also include, in plain text:

- SDR model, antenna type and rough placement (indoor/outdoor, height);
- environment class (rural / suburban / urban / industrial / event site);
- for drone-labelled rows: airframe and control-link type if you know it
  (e.g. "generic 2.4 GHz FHSS toy", "ELRS 915", "OcuSync-class"),
  approximate distance and whether it was flying or idling on the ground;
- anything you knew was transmitting nearby (Wi-Fi APs, BLE beacons).

**Labels matter more than volume.** Fifty honestly-labelled observations
from a known drone beat fifty thousand unlabelled ones.

### 2. Negative-only datasets - also valuable

A dataset with *no* drone rows, recorded at a site with no drone
present, is still useful: it measures the false-alarm rate somewhere
other than the single environment the defaults were tuned in. Same
command, just label the interesting events `not_drone`.

### 3. Baseline environments - useful, but read the privacy note

A `data/environments/<name>/` directory is a per-frequency-bin noise
profile of your site. It helps calibrate what "normal" looks like
elsewhere - but it is also a fingerprint of your local RF environment.
Share it only if you are comfortable with that, and prefer a site you do
not live at.

### 4. IQ captures - only under the conditions below

Short triggered captures (`sweep.capture.enabled: true`) are the richest
data, and the most sensitive. Please only send IQ that contains
**transmissions from equipment you own or are authorised to test**,
recorded deliberately for that purpose.

## What NOT to send

- **Anything containing third-party communications.** In many
  jurisdictions intercepting, recording, or passing on radio traffic you
  are not a party to is a criminal offence, regardless of intent, and no
  research goal makes that acceptable. drone4rf never demodulates
  content - please do not use other tools to add any.
- Raw `events.db` files. They carry timestamps of every detection, which
  is effectively an occupancy log of the building they were recorded in.
  Export a feature dataset instead.
- Anything that identifies a specific person, household or address.
  "Suburban UK, 2-storey house, dipole in the loft" is the right level of
  detail; a street address is not.
- Captures made to monitor a particular neighbour, vehicle or person.
  That is surveillance, not calibration, and it is not welcome here.

## How to send it

Open an issue on this repository describing what you have, and we will
agree a transfer method - please **do not attach IQ captures or
databases to a public issue** without checking first.

<!-- Optional: add a contact address here if you would rather receive
     data privately, e.g.  Private contact: your-address@example.com -->

## How contributed data will be used

- To measure detection/false-alarm rates across sites, and to re-fit the
  fusion weights in `analytics.weights` against something broader than a
  single environment.
- To train reference models. Any published model will state the sites,
  hardware and class balance it was trained on, and will keep the
  existing calibration and unknown-class rejection behaviour.
- Contributors are credited unless they ask not to be. Say so in the
  issue if you would prefer anonymity.
- Nothing is redistributed without asking you first.

## A known limitation of the current schema

Feature vectors deliberately exclude absolute frequency, which is good
for privacy but limits per-band signature work. A future feature-schema
version will likely add *band-relative* descriptors (offset within the
band, channel spacing) rather than raw frequencies. Feedback on that
trade-off is welcome - the schema is versioned, and models refuse to
load across a version change, so it can be revised safely.

---

## Code contributions

```sh
python -m venv .venv && .venv/Scripts/activate   # Linux: source .venv/bin/activate
pip install -e .[dev]
pytest
```

The whole suite runs without any SDR hardware - simulated and file-based
sources cover the pipeline end to end. Please keep it that way: new
features need tests that pass on a machine with no radio attached.

House rules that matter more than style:

- **Receive-only.** No transmit path, ever. Pull requests adding TX,
  jamming, spoofing, or packet injection will be closed.
- **No content demodulation.** Spectral statistics and timing features
  only.
- **No overclaiming.** The confidence vocabulary is capped at
  "high-confidence drone-related activity"; "confirmed" does not exist
  and should not be added. Detectors return explanations, not verdicts.
