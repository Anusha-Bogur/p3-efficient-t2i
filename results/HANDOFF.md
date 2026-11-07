##### \# HANDOFF for P4 

##### \## Live URL

##### \- URL: https://huggingface.co/spaces/anu-122006/p3-efficient-t2i

##### \- Model config running in the app: bk\_final

##### \- Tested on phone: yes | laptop: yes | first-click time: about 1.1 s generation on the live Space (Low mode, seed 0), plus page load

##### \## Adaptation (P1)

##### \- (P1 to fill in)

##### \## Results (P2) -> results/summary.csv, results/quality\_vs\_cost.png

##### \- (P2 to fill in)

##### \## Failure cases (P3) -> results/failures/

##### 1\. prompt: a red car beside a blue house (Compositional, layout "col") | what went wrong: car is red and on the left, but the house wall came out mostly red; blue appears only on the door and planter | likely cause: the text encoder does not reliably tie each colour to its own object, so Compositional and Medium only help partly | files: colour\_compositional.webp, colour\_medium.webp

##### 2\. prompt: a cup to the left of a plate | what went wrong: object positions are not reliably followed | likely cause: the text encoder does not encode left/right relations well | file: spatial.webp

##### 3\. prompt: a sign that says OPEN | what went wrong: the sign text is not rendered correctly | likely cause: the text encoder and distilled UNet do not render letters reliably | file: text.webp

##### \## Anything P4 must know

##### \- Demo hosted on a Hugging Face ZeroGPU Space (free shared GPU); reported latencies were measured on an NVIDIA T4.

##### \- Log in to Hugging Face before using the demo, otherwise the GPU quota can run out ("quota reached" is not a bug).

##### \- Three modes: Low (standard bk\_final, 20 steps), Compositional (splits side-by-side prompts into regions; prompts without a multi-object layout fall back to a normal image), Medium (3 candidates at 25 steps, CLIP picks the best).

##### \- Timings on the live Space (my runs, not the evaluation numbers): Low 1.14 s with CV and 1.09 s without; Compositional 1.22 s and 1.28 s; Medium 2.18 s. P1's code comments give T4 timings of about 1.9-2.2 s (Low), 3.5 s (Compositional) and 7.8 s (Medium). Use P2's table for any reported latency.

##### \- The classical CV step keeps a correction only if the quality score improves. Example: on the pizza and orange juice image the score dropped from 0.9165 to 0.7768, so the correction was reverted (correction\_kept: false).

##### \- Checks passed: empty prompt shows the "Please enter a prompt." popup; CV toggle off gives "classical\_cv": "off"; works on phone and laptop.

##### \- Evidence images and reports: results/demo/ and results/failures/.

