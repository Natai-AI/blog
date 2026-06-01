# Video is the next frontier, and your VLM has amnesia

*on why a "memory layer" is the missing piece for video-native AI — and how I built one*

---

I want to start with an observation that took me embarrassingly long to internalize: **VLMs are brilliant and they have no memory.**

You hand a vision-language model a frame and it will tell you, in eerily good prose, that there's a family at a dinner table, someone's reaching for the salt, the lighting is warm, the mood is convivial. Magic. Then you hand it the next frame and it has *no idea* it ever saw the first one. It's the world's most talented intern with anterograde amnesia. Every frame is the first frame. Every clip is a blank slate.

For images this doesn't matter. For *video* it's everything. Video is not a bag of frames — it's a sequence with identity ("the person who walked in at 0:03 is the same person sitting down at 0:08"), with causality, with stuff that happens *off-screen-now-but-mattered-then*. The thing that makes video video is exactly the thing the VLM throws away between forward passes.

So how does everyone deal with this today? Two bad options.

**Option A: stuff everything into context.** Sample frames, base64 them, dump them all into the prompt. This works for a 10-second clip and detonates for anything real. A 20-minute video at even 1 fps is 1,200 images. You will run out of context window, money, and patience, roughly in that order. And even if you could fit it, you're making the model re-derive "who is this person" from pixels on every single call. Insane.

**Option B: caption everything up front.** Run a VLM over the video, get a wall of text, query the text. Better! But you've now thrown away all the *structure* — the precise timestamps, the bounding boxes, the fact that object `t0` in shot 1 is the same object as `t0` in shot 5. You've compressed a structured world into prose and lost the handles you need to actually *do* anything.

The punchline of this post is that there's an Option C, and it's the same trick that fixed this exact problem for text two years ago.

## Retrieval, but for video

For LLMs we figured out that you don't put the whole knowledge base in context — you **retrieve** the relevant slice and inject it. RAG. Obvious in hindsight.

Video needs the same thing, but the "documents" are weirder. The relevant unit isn't a paragraph, it's a *shot*. And a shot isn't just text — it's a little structured world: who's in it, where they are (boxes), how they move (tracks), what it sounds like (transcript), and what it *looks like* in a way you can compare to other shots (an embedding).

So I built a thing called **deja-reve** — a video memory layer. The whole idea fits in one sentence:

> Process the video once into a structured, queryable memory; then at question-time, retrieve only the relevant shots and hand the VLM a compact context block instead of the raw video.

It's RAG where the chunks are shots and the embeddings are visual. Let me show you what that actually looks like, because the concreteness is the point.

## What the memory looks like

You ingest a video once:

```bash
uv run deja-reve ingest dinner.mp4 --db dinner.db
```

Under the hood this runs a little pipeline — scene detection to cut the video into shots, an object detector (RF-DETR) with a tracker to find and *follow* things across frames, a vision backbone (DINOv2) to produce one embedding per shot, and an ASR pass for speech. All of it lands in a single SQLite file with `sqlite-vec` for vector search. No cloud, no API keys, runs on your laptop.

Now the good part. You ask a question, it retrieves the relevant shot, and it hands the VLM *this*:

```
VIDEO CONTEXT:
- Shot #1: 4.0s - 8.0s (duration: 4.0s)
- Objects in frame: person (t0), chair (t1), chair (t2), person (t3), dining table (t-1)
  - t0 at [536, 408, 757, 923]
  - t3 at [768, 458, 1120, 1063]
- Visual: bottle, bowl, chair, cup, dining table, person, wine glass
- Previous shot: #0 (0.0s - 4.0s)
- Next shot: #2 (8.0s - 10.0s)
```

Stare at that for a second, because there's a lot of quiet genius in how *little* it is.

It's a few hundred tokens. It tells the VLM the timestamps (so it can reason temporally), the objects *and their boxes* (so it can reason spatially — "the person on the left"), a track id `t0` that is **stable across shots** (so "the person who sat down" is a thing you can follow), and pointers to the neighboring shots (so "what happened just before" is one hop away). It is the structured world, losslessly enough to act on, cheaply enough to afford.

This is the format the VLM reads. The `format_shot_context()` function that produces it *is* the API. Everything else is plumbing.

## Using it: context for a VLM

The loop is dead simple:

```python
from deja_reve.db import MemoryDB
from deja_reve.retrieval import RetrievalEngine
from deja_reve.prompt import format_shot_context

db = MemoryDB("dinner.db")
engine = RetrievalEngine(db)

# 1. retrieve the relevant slice (here: around the 5s mark)
result = engine.query_time_range(4.0, 6.0)

# 2. format it as a context block
context = format_shot_context(result)

# 3. inject it as ground truth, then ask
prompt = f"""{context}

Use the VIDEO CONTEXT above as ground-truth observations.
Q: What's happening at the dinner table around the 5-second mark?
"""
answer = your_vlm(prompt)   # optionally + the actual frame image
```

That's it. You can retrieve by time, by shot id, by *visual similarity* ("find shots that look like this one"), by mood shift ("where does the scene change"), or by transcript search. The VLM stops guessing about a video it can't see and starts reasoning over facts that are actually in the video.

And here's the kicker I didn't expect until I used it: **the boxes and track ids make the VLM dramatically less hallucination-prone.** When you tell it "there are two people, here are their coordinates," it stops inventing a third. Grounding isn't just nice, it's load-bearing.

## Where it gets genuinely fun: editing

Once the VLM can *see* the video as structure, it can *edit* it. The mental model:

> **deja-reve (memory) -> VLM (decisions) -> ffmpeg (execution).**

The VLM never cuts video. It reads the context and emits an edit decision list — JSON that references real timestamps and real boxes from the DB. Then dumb, reliable ffmpeg does the cutting.

"Make me a 15-second vertical highlight reel, keep only shots with people at the table, blur the faces" becomes: the VLM picks shot ids and timestamps *that already exist in the memory*, builds crop windows *from the bounding boxes deja-reve stored*, and you hand the result to ffmpeg. The edit is precise because every decision is anchored to something real. The VLM is the director; it is not holding the scissors.

The table of what-feeds-what writes itself:

| Editing task | What deja-reve provides |
|---|---|
| frame-accurate cuts | shot timestamps |
| auto-reframe to vertical | bounding boxes |
| follow one subject across cuts | track ids |
| drop redundant b-roll | visual embeddings |
| chapter boundaries | mood shifts |
| caption / cut on dialogue | transcripts |

## A confession about the embeddings (this part matters)

I want to be honest about a bug I shipped, because it's instructive.

The first version stored the raw DINOv2 embeddings and searched them with L2 distance. Seemed fine. It was not fine. Mean-pooled backbone features carry *magnitude* that's driven by brightness and texture density, not content — so two identical scenes at different exposures looked far apart, and any similarity threshold I tuned on one video silently broke on the next.

The fix is one line of "duh": **L2-normalize the vectors and use cosine distance.** On unit vectors, distance becomes content similarity, bounded on a clean 0->2 scale, and a threshold of 0.4 means the same thing on every video. On my dinner clip the distances went from an uninterpretable `6.50` to a legible `0.019` ("basically the same shot," which is correct).

I mention this because it's the whole thesis in miniature: the magic of these systems is real, but it lives or dies on boring, unglamorous representational hygiene. Normalize your embeddings. Map your class ids correctly (yes, I shipped an off-by-one COCO mapping that turned a family dinner into "bicycles and TVs" — the model was right, my lookup table was wrong). Gate your ASR so it doesn't hallucinate "8, 8, 8, 8" over silence. The model is the easy part now. The plumbing is where you earn it.

## The thing I actually believe

Text got a memory layer and it unlocked a Cambrian explosion of agents. Video is sitting exactly where text was: phenomenal per-frame intelligence, zero persistence, no handles to grab. The bottleneck is not the model. The model is *fine*. The bottleneck is that we keep asking an amnesiac to remember.

Give it a memory — structured, retrievable, grounded in timestamps and boxes and stable identities — and the VLM stops being a frame classifier and starts being something that can reason about, search through, and *edit* video. That's the unlock. That's why I think a boring SQLite file is, unironically, the revolutionary part.

deja-reve is my small bet on that. It's local, it's a few hundred lines, and the entire interface is "give the model a good paragraph about what it's looking at." Turns out that's most of the battle.

```bash
uv run deja-reve ingest your_video.mp4 --db memory.db
uv run deja-reve query --db memory.db --shot 1
```

Go give your VLM a memory.
