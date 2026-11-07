\## 1. Colour leakage

\- Prompt: a red car parked beside a blue house

\- Seed: 0

\- What went wrong: the house is red instead of blue

\- Likely cause: the text encoder doesn't reliably tie each colour to its own object

\- Extra finding: classical CV (white balance) tinted the red-dominated scene toward cyan (colour\_after\_cv.webp), even though the quality score rose from 0.9484 to 0.9755. The quality score does not measure colour balance.



\## 2. Spatial relation

\- Prompt: a cup to the left of a plate

\- Seed: 0

\- What went wrong: the cup sits inside the bowl instead of to the left of a plate

\- Likely cause: the text encoder doesn't reliably bind positions to the image



\## 3. Text in image

\- Prompt: a sign that says OPEN

\- Seed: 0

\- What went wrong: the sign has garbled, unreadable pseudo-letters; "OPEN" is not spelled anywhere

\- Likely cause: the model isn't trained to draw readable letters, and the text encoder doesn't pass exact spellings to the image



\### Colour binding: "a red car beside a blue house" (Compositional)

\- Layout "col" was applied (regions: a red car, a blue house).

\- Result: car red and on the left, but the house wall came out mostly red; blue appears only on the door and planter.

\- Cause: the text encoder does not reliably tie each colour to its own object, so Compositional and Medium only help partly.

