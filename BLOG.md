# Your VLM has amnesia

*why a "memory layer" is the missing piece for video-native AI, and how we built one*

---

I want to start with an observation that took me embarrassingly long to internalize: **VLMs are brilliant and they have no memory.**

You hand a vision language model a frame and it will tell you, with eerie accuracy, that there's a family at a dinner table, someone's reaching for the salt, the lighting is warm, the vibes are cozy. Magic. Then you hand it the next frame and it has *no idea* what it saw in the first one. It's the world's most talented intern with anterograde amnesia. Every frame is the first frame. Every clip is a blank slate.

For images this doesn't matter. For *video* it's everything. Video is not a bag of frames, it's a sequence with identity ("the person who walked in at 0:03 is the same person sitting down at 0:08"), with causality, with stuff that happens *off screen now but mattered then*. The thing that makes video, video, is exactly the thing the VLM throws away between forward passes.

So how does everyone deal with this today? Two bad options.

**Option A: stuff everything into context.** Sample frames, base64 them, dump them all into the prompt. This works for a 3 to 10 second clip and detonates for anything else. A 2.5 minute clip at 1 fps is 143 images. And even if you could afford it, the model has to re-derive "who is this person" from pixels on every call. No timestamps, no bounding boxes, no identity across frames. Just an expensive bag of pixels.

**Option B: caption everything up front.** Run a VLM over the video, get a wall of text, query the text. Better! But you've thrown away all the *structure*  the precise timestamps, the bounding boxes, the fact that object `t0` in shot 1 is the same object as `t0` in shot 5. You've compressed a structured world into prose and lost the handles you need to actually *do* anything.

Here's the punchline: there's an Option C, and it's the same trick that fixed this exact problem for text two years ago!

## Retrieval, but for video

For LLMs we figured out that you don't put the whole knowledge base in context. You **retrieve** the relevant slice and inject it. RAG. Obvious in hindsight.

Video needs the same thing, but the "documents" are weirder. The relevant unit isn't a paragraph, it's a *shot*. And a shot isn't just text it's a little structured world: who's in it, where they are (boxes), how they move (tracks), what it sounds like (transcript), and what it *looks like* in a way you can compare to other shots (an embedding).

So we built a thing called **deja-reve**, a video memory layer. The whole idea fits in one sentence:

> Process the video once into a structured, queryable memory. Then at question time, retrieve only the relevant shots and hand the VLM a compact context block instead of the raw video.

It's RAG where the chunks are shots and the embeddings are visual. Let me show you what that actually looks like, because the concreteness is the point.

## What the memory looks like

You ingest a video once:

```bash
deja-reve ingest dinner.mp4 --db dinner.db
```

Under the hood this runs a pipeline: scene detection cuts the video into shots, an object detector (RF-DETR) with a tracker finds and *follows* things across frames, a vision backbone (DINOv2) produces one embedding per shot, and an ASR pass captures speech. All of it lands in a single SQLite file with `sqlite-vec` for vector search. No cloud, no API keys, runs on your laptop.

On a 2.5 minute dinner scene at 1080p, the ingestion produces: 22 shots, 17 tracked objects, 270 bounding box observations, 6 speech transcripts, 22 visual embeddings. The database is 1.6 MB.

Now the good stuff. You ask a question, it retrieves the relevant shot, and it hands the VLM *this*:

```
VIDEO CONTEXT:
- Shot #3: 20.6s - 24.4s (duration: 3.8s)
- Objects in frame: person (t0), person (t2), chair (t4), person (t5), dining table (t-1)
  - t0 at [1006, 36, 1919, 1050]
  - t2 at [436, 410, 942, 1044]
  - t4 at [25, 623, 646, 1042]
  - t5 at [420, 367, 715, 680]
  - t-1 at [914, 580, 1121, 1045]
- Visual: bottle, bowl, chair, cup, dining table, handbag, person, spoon
- Previous shot: #2 (14.5s - 20.6s)
- Next shot: #4 (24.4s - 26.0s)
```

That's 111 tokens. Stare at it for a second, because there's a lot of quiet genius in how *little* it is.

## Show me the numbers

Every claim in that context block is something no other approach provides. I benchmarked this on a real 2.5 minute, 1080p dinner video. Here's what each approach gives the VLM:

**Timestamps.** deja-reve: every shot has precise `start_sec`/`end_sec` (20.6s–24.4s). Frame-stuffing: frame indices only, the VLM can't map to seconds. Pre-captioning: timestamps only if the captioner includes them, which most don't.

**Bounding boxes.** deja-reve: pixel precise coordinates for every object, `t0 at [1006, 36, 1919, 1050]` means person t0 occupies the right half of the frame. 270 observations stored across 22 shots. Frame-stuffing: zero spatial metadata. The VLM has to re-detect everything from pixels, and it can't hand those detections to a downstream tool. Pre-captioning: "a person on the right", prose, not coordinates. You can't crop a video with prose.

**Stable track IDs.** This is the killer feature. deja-reve tracks 6 people across the entire video with persistent IDs:

| Track | Shots | Time span |
|---|---|---|
| t10 | 11 shots | 51s → 99s (48 seconds) |
| t0 | 8 shots | 9s → 86s (77 seconds) |
| t13 | 5 shots | 99s → 133s (34 seconds) |
| t2 | 5 shots | 9s → 33s (25 seconds) |

Person `t0` appears in shot #1 and is *the same* `t0` in shot #11, seventy seven seconds later. A VLM without track IDs has no mechanism for this  "a person" in shot 1 is a completely different token from "a person" in shot 5. The question "who sat down?" is literally unanswerable without stable identity across shots. Neither frame stuffing nor captioning provides this. Only tracking does.

**Neighboring shot pointers.** Every context block includes `Previous shot: #2` and `Next shot: #4`. "What happened just before the cup dropped?" is one database hop. Without this, frames are an unlinked bag and captions are independent paragraphs.

**Cheap enough to afford.** Here's the cost comparison on this video:

| Approach | Tokens per query |
|---|---|
| Frame-stuffing (1 fps) | 502,931 |
| All captions in context | 1,100 |
| **deja-reve (single shot)** | **~100** |

deja-reve is **~5,000× cheaper** than frame stuffing per query. And it provides *more* information: structured, queryable facts vs. raw pixels the model has to re-interpret every time.

**Lossless enough to act on.** Those bounding boxes aren't decorative. `t0 at [1006, 36, 1919, 1050]` is a crop window ffmpeg can execute. The timestamps are frame accurate cut points. The track IDs tell you which object to follow across shots. Every field in the context block is a handle a downstream tool can grab.

## The grounding effect: why this kills hallucination

Here's the claim I want to be precise about: **the boxes and track IDs make the VLM dramatically less hallucination prone.**

Research shows VLMs hallucinate object counts 20–40% of the time when working from pixels alone (Li et al. 2023, *Evaluating Object Hallucination in Large Vision Language Models*). Ask "how many people are at the table?" and the model will frequently add or miss people, especially with occlusion or unusual poses.

Here's what changes with deja-reve. Shot #3 of the dinner video has three people. Without grounding, the VLM sees 3,517 tokens of pixels and has to guess:

```
Without deja-reve:
  Input: 3,517 tokens (one 1080p frame)
  VLM must: re-detect all objects from pixels
  VLM gets: no confidence scores, no coordinates, no identity
  Risk:     hallucinate extra people, miss occluded ones,
            lose identity across shots
```

With deja-reve, the VLM reads structured facts:

```
With deja-reve:
  Input: 111 tokens
  VLM reads: "3 people: t0 at [1006, 36, 1919, 1050],
              t2 at [436, 410, 942, 1044],
              t5 at [420, 367, 715, 680]"
  VLM gets: exact count, pixel coordinates, stable IDs
  Result:   it doesn't guess - it reads. No hallucination.
```

When you tell the model "there are three people, here are their coordinates and IDs," it stops inventing a fourth. It doesn't need to guess the count  the count is stated. It doesn't need to guess positions, the boxes are given. It doesn't need to guess identity — the track IDs are stable. Grounding isn't just nice. It's load bearing.

And this compounds across shots. Without track IDs, the VLM sees "a person" in shot #3 and "a person" in shot #14 and has zero signal they're the same person. With deja-reve, it sees `t0` in both: same ID, same individual, eleven shots and sixty three seconds apart. Cross shot reasoning goes from impossible to trivial.

## Using it: context for a VLM

The loop is dead simple:

```python
from deja_reve.db import MemoryDB
from deja_reve.retrieval import RetrievalEngine
from deja_reve.prompt import format_shot_context

db = MemoryDB("dinner.db")
engine = RetrievalEngine(db)

# 1. retrieve the relevant slice
result = engine.query_time_range(20.0, 25.0)

# 2. format it as a context block
context = format_shot_context(result)

# 3. inject it as ground truth, then ask
prompt = f"""{context}

Use the VIDEO CONTEXT above as ground truth observations.
Q: What's happening at the dinner table around the 22 second mark?
"""
answer = your_vlm(prompt)   # optionally + the actual frame image
```

That's it. You can retrieve by time, by shot ID, by *visual similarity* ("find shots that look like this one"), by mood shift ("where does the visual tone change"), or by transcript search. The VLM stops guessing about a video it can't see and starts reasoning over facts that are actually in the video.

## Where it gets genuinely fun: editing

Once the VLM can *see* the video as structure, it can *edit* it. The mental model:

> **deja-reve (memory) → VLM (decisions) → ffmpeg (execution)**

The VLM never cuts video. It reads the context and emits an edit decision list — JSON that references real timestamps and real boxes from the DB. Then dumb, reliable ffmpeg does the cutting.

"Make me a 15 second vertical highlight reel, keep only shots with people at the table, blur the faces" becomes: the VLM picks shot IDs and timestamps *that already exist in the memory*, builds crop windows *from the bounding boxes deja-reve stored*, and you hand the result to ffmpeg. The edit is precise because every decision is anchored to something real. The VLM is the director; it is not holding the scissors.

| Editing task | What deja-reve provides |
|---|---|
| frame-accurate cuts | shot timestamps (20.6s–24.4s) |
| auto-reframe to vertical | bounding boxes ([1006, 36, 1919, 1050]) |
| follow one subject across cuts | track IDs (t0 across 8 shots, 77 seconds) |
| drop redundant b-roll | visual embeddings (384-dim DINOv2, KNN search) |
| chapter boundaries | embedding deltas between adjacent shots |
| caption / cut on dialogue | speech transcripts aligned to shots |

## Confession time

I want to be honest about a bug we shipped, because it's instructive.

The first version stored the raw DINOv2 embeddings and searched them with L2 distance. Seemed fine. It was not fine. Mean pooled backbone features carry *magnitude* driven by brightness and texture density, not content. Two identical scenes at different exposures looked far apart, and any similarity threshold tuned on one video silently broke on the next.

The fix is simple: **L2-normalize the vectors and use cosine distance.** On unit vectors, distance becomes content similarity, bounded on a clean 0→2 scale, and a threshold of 0.4 means the same thing on every video. On the dinner clip the distances went from an uninterpretable `6.50` to a legible `0.019` ("basically the same shot," which is correct).

I mention this because it's the whole thesis in miniature: the magic of these systems is real, but it lives or dies on boring, unglamorous representational hygiene. Normalize your embeddings. Map your class IDs correctly (yes, I shipped an off by one COCO mapping that turned a family dinner into "bicycles and TVs"  the model was right, my lookup table was wrong). Gate your ASR so it doesn't hallucinate "8, 8, 8, 8" over silence. The model is the easy part now. The plumbing is where you earn it.

## The thing I actually believe

Text got a memory layer and it unlocked a Cambrian explosion of agents. Video is sitting exactly where text was: phenomenal per frame intelligence, zero persistence, no handles to grab. The bottleneck is not the model. The model is *fine*. The bottleneck is that we keep asking an amnesiac to remember.

Give it a memory. Structured, retrievable, grounded in timestamps and boxes and stable identities. Then the VLM stops being a frame classifier and starts being something that can reason about, search through, and *edit* video. That's the unlock. That's why I think a boring SQLite file is, unironically, the revolutionary part.

deja-reve is our small bet on that. It's local, it's a few hundred lines, and the entire interface is "give the model a good paragraph about what it's looking at." Turns out that's most of the battle.

## Install 

```bash
pip install deja-reve
```

## Usage
```bash
deja-reve ingest your_video.mp4 --db memory.db
deja-reve query --db memory.db --shot 1
```

## Repo
https://github.com/Natai-AI/deja-reve


Go give your VLM something to remember.
